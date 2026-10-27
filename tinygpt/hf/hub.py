# coding: utf-8
'''
Fetch a model from the Hugging Face Hub -- just the files inference needs
-- into a local cache that works with the network switched off.

No huggingface_hub dependency: one metadata call to the public REST API,
then plain HTTPS downloads with `requests`.

    path = snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct')
    path = find_local('HuggingFaceTB/SmolLM2-135M-Instruct')   # offline

What is downloaded is an allowlist: config.json, generation_config.json,
the tokenizer files and chat template, and safetensors weights (one file, or shards named
by model.safetensors.index.json). Nothing executable is fetched, and a
repo that only offers pickled PyTorch weights (*.bin, *.pt) is refused
rather than loaded: unpickling runs arbitrary code.

Every file is checked on completion against what the Hub says it should
be -- the SHA-256 for LFS files (the weights), the git blob SHA-1 for
small files -- so a truncated or corrupted download can never be loaded.
Interrupted downloads resume from the partial file with an HTTP Range
request when the server supports it.

Cache layout, under cache_dir:

    <org>--<name>/refs/<revision>        the commit a revision resolved to
    <org>--<name>/<commit>/<files>       the snapshot
    <org>--<name>/<commit>/manifest.json written last; marks it complete

`requests` is imported only when something is actually downloaded, so
the offline paths (find_local, import_local) work without it.
'''

import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile

ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')

REQUIRED = ('config.json', 'tokenizer.json')
OPTIONAL = ('generation_config.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'chat_template.jinja')
SINGLE = 'model.safetensors'
INDEX = 'model.safetensors.index.json'
SHARD_RE = re.compile(r'^model-\d{5}-of-\d{5}\.safetensors$')
UNSAFE_WEIGHTS = ('.bin', '.pt', '.pth', '.ckpt', '.pkl', '.pickle')
REPO_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,95}'
                     r'/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')
REV_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$')

CHUNK = 1 << 20          # hashing / copying
STREAM_CHUNK = 1 << 18   # network reads: progress lands in .part often


class HubError(RuntimeError):
    pass


def default_cache_dir():
    '''hf_models/ next to the repository's scripts (in Pythonista: inside
    the Documents folder, so it is visible in the file browser), unless
    TINY_GPT_HF_CACHE says otherwise.'''
    env = os.environ.get('TINY_GPT_HF_CACHE')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', 'hf_models'))


_URL_RE = re.compile(r'^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/'
                     r'([^/?#]+/[^/?#]+)(?:/(?:tree|blob|resolve)/[^?#]*)?'
                     r'/?(?:[?#].*)?$')


def normalize_repo_id(text):
    '''Accept 'owner/name' or a model page URL as copied from a browser
    (https://huggingface.co/owner/name, optionally .../tree/main), and
    return 'owner/name'. Anything else is returned unchanged for
    check_repo_id() to reject.'''
    text = (text or '').strip()
    m = _URL_RE.match(text)
    return m.group(1) if m else text


def check_repo_id(repo_id):
    if not REPO_RE.match(repo_id or '') or '..' in repo_id:
        raise HubError('not a model repository id: %r (expected '
                       '"owner/name")' % (repo_id,))
    return repo_id


def check_filename(name):
    '''Only plain file names from the allowlist; never a path.'''
    if (not name or name != os.path.basename(name) or name.startswith('.')
            or '\\' in name or '\0' in name):
        raise HubError('refusing unexpected file path %r' % (name,))
    if not (name in REQUIRED or name in OPTIONAL or name in (SINGLE, INDEX)
            or SHARD_RE.match(name)):
        raise HubError('refusing file %r: not on the allowlist' % (name,))
    return name


def repo_dir(repo_id, cache_dir=None):
    check_repo_id(repo_id)
    return os.path.join(cache_dir or default_cache_dir(),
                        repo_id.replace('/', '--'))


# ----------------------------------------------------------------- hashing

def sha256_file(path, progress=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path):
    size = os.path.getsize(path)
    h = hashlib.sha1(b'blob %d\0' % size)
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_file(path, spec):
    '''Check a file against a manifest/Hub entry; raise HubError if not.'''
    if not os.path.exists(path):
        raise HubError('%s is missing' % path)
    size = os.path.getsize(path)
    if spec.get('size') is not None and size != spec['size']:
        raise HubError('%s is %d bytes, expected %d'
                       % (path, size, spec['size']))
    if spec.get('sha256'):
        got = sha256_file(path)
        if got != spec['sha256']:
            raise HubError('%s: sha256 %s, expected %s'
                           % (path, got, spec['sha256']))
    elif spec.get('git_sha1'):
        got = git_blob_sha1(path)
        if got != spec['git_sha1']:
            raise HubError('%s: git sha1 %s, expected %s'
                           % (path, got, spec['git_sha1']))


# ---------------------------------------------------------------- progress

class Progress(object):
    '''Prints a line roughly every `every` seconds; Pythonista-friendly
    (no carriage-return tricks, which its console does not honour).'''

    def __init__(self, every=2.0, out=None):
        self.every = every
        self.out = out or sys.stdout
        self._last = 0.0
        self._t0 = None
        self._shown = None

    def __call__(self, name, done, total, final=False):
        now = time.time()
        if self._t0 is None or done == 0:
            self._t0 = now
        if not final and now - self._last < self.every:
            return
        if final and self._shown == (name, done):
            return
        self._last = now
        self._shown = (name, done)
        rate = done / max(now - self._t0, 1e-6) / 1e6
        if total:
            self.out.write('  %-28s %7.1f / %.1f MB  %5.1f%%  %.1f MB/s\n'
                           % (name, done / 1e6, total / 1e6,
                              100.0 * done / total, rate))
        else:
            self.out.write('  %-28s %7.1f MB  %.1f MB/s\n'
                           % (name, done / 1e6, rate))
        self.out.flush()


# ------------------------------------------------------------------ remote

def _session(token=None):
    try:
        import requests
    except ImportError:
        raise HubError('downloading needs the `requests` package')
    s = requests.Session()
    token = token or os.environ.get('HF_TOKEN')
    if token:
        s.headers['Authorization'] = 'Bearer ' + token
    s.headers['User-Agent'] = 'tiny_gpt-hf/1.0'
    return s


def repo_info(repo_id, revision='main', session=None, endpoint=ENDPOINT):
    '''{'sha': commit, 'files': {name: {'size', 'sha256'|'git_sha1'}}}.'''
    check_repo_id(repo_id)
    if not REV_RE.match(revision) or '..' in revision:
        raise HubError('bad revision %r' % (revision,))
    s = session or _session()
    url = '%s/api/models/%s/revision/%s' % (endpoint, repo_id, revision)
    r = s.get(url, params={'blobs': 'true'}, timeout=(15, 60))
    if r.status_code == 401 or r.status_code == 403:
        raise HubError('%s: access denied (HTTP %d); gated or private '
                       'repos need HF_TOKEN' % (repo_id, r.status_code))
    if r.status_code == 404:
        raise HubError('%s@%s: not found on the Hub' % (repo_id, revision))
    r.raise_for_status()
    d = r.json()
    sha = d.get('sha')
    if not sha or not re.match(r'^[0-9a-f]{40}$', sha):
        raise HubError('Hub returned no commit sha for %s' % repo_id)
    files = {}
    for sib in d.get('siblings') or ():
        name = sib.get('rfilename')
        if not name:
            continue
        spec = {'size': sib.get('size')}
        lfs = sib.get('lfs')
        if lfs:
            spec['sha256'] = lfs.get('sha256')
            spec['size'] = lfs.get('size', spec['size'])
        elif sib.get('blobId'):
            spec['git_sha1'] = sib['blobId']
        files[name] = spec
    return {'sha': sha, 'files': files}


def select_files(files):
    '''The names to fetch from a repo listing, or HubError explaining why
    the repo cannot be used.'''
    top = set(n for n in files if '/' not in n)
    missing = [n for n in REQUIRED if n not in top]
    if missing:
        raise HubError('repo lacks %s; only models with a config.json and a '
                       'tokenizer.json are supported' % ', '.join(missing))
    if SINGLE not in top and INDEX not in top:
        unsafe = sorted(n for n in top if n.endswith(UNSAFE_WEIGHTS))
        if unsafe:
            raise HubError('repo has only pickled weights (%s); those can run '
                           'arbitrary code when loaded and are refused. '
                           'Only safetensors are supported.'
                           % ', '.join(unsafe[:3]))
        raise HubError('repo has no safetensors weights')
    want = list(REQUIRED) + [n for n in OPTIONAL if n in top]
    want.append(INDEX if INDEX in top else SINGLE)
    return want


def shards_from_index(path):
    with open(path, 'rb') as f:
        idx = json.loads(f.read().decode('utf-8'))
    wm = idx.get('weight_map')
    if not isinstance(wm, dict) or not wm:
        raise HubError('%s has no weight_map' % path)
    shards = sorted(set(wm.values()))
    for s in shards:
        if not SHARD_RE.match(s or ''):
            raise HubError('index names unexpected shard file %r' % (s,))
    return shards


def _short(e):
    # requests errors embed the full (signed, very long) CDN URL.
    msg = re.sub(r'(https?://[^/\s]+)/\S+', r'\1/...', str(e))
    msg = re.sub(r'(url: )/\S+', r'\1/...', msg)
    return '%s: %s' % (e.__class__.__name__, msg[:300])


def download_file(session, url, dest, spec, progress=None, retries=3):
    '''Stream url to dest, resuming a dest.part left by an earlier try.

    The file only appears at `dest` once its size and hash check out.
    '''
    part = dest + '.part'
    total = spec.get('size')
    name = os.path.basename(dest)
    for attempt in range(retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and have > total:
            os.remove(part)
            have = 0
        headers = {}
        if have and (total is None or have < total):
            headers['Range'] = 'bytes=%d-' % have
        try:
            if total is not None and have == total:
                pass        # fully downloaded already; just verify
            else:
                with session.get(url, headers=headers, stream=True,
                                 timeout=(15, 60)) as r:
                    if r.status_code == 416:
                        # Range not satisfiable: the part is unusable.
                        os.remove(part)
                        continue
                    r.raise_for_status()
                    mode = 'ab'
                    if have and r.status_code != 206:
                        # The server ignored Range; start over.
                        have, mode = 0, 'wb'
                    done = have
                    with open(part, mode) as f:
                        for b in r.iter_content(STREAM_CHUNK):
                            if b:
                                f.write(b)
                                done += len(b)
                                if progress:
                                    progress(name, done, total)
                    if progress:
                        progress(name, done, total, final=True)
                    if total is not None and done < total:
                        # Some stacks end a dropped stream quietly rather
                        # than raising; that is an interruption, so resume.
                        raise IOError('connection closed at %d of %d bytes'
                                      % (done, total))
            verify_file(part, spec)
            os.replace(part, dest)
            return dest
        except HubError:
            # Complete but wrong: never resume from a corrupt part.
            if os.path.exists(part):
                os.remove(part)
            if attempt == retries:
                raise
        except Exception as e:      # network trouble: retry, resuming
            if attempt == retries:
                raise HubError('download of %s failed: %s' % (name,
                                                              _short(e)))
            wait = 2 ** attempt
            sys.stdout.write('  %s: %s; retrying in %ds\n'
                             % (name, e.__class__.__name__, wait))
            time.sleep(wait)
    raise HubError('download of %s failed' % name)


def _write_manifest(snap, repo_id, commit, specs, source):
    man = {'repo_id': repo_id, 'commit': commit, 'source': source,
           'files': specs, 'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    tmp = os.path.join(snap, 'manifest.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(man, f, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(snap, 'manifest.json'))


def _write_ref(rdir, revision, commit):
    refs = os.path.join(rdir, 'refs')
    path = os.path.join(refs, revision.replace('/', '--'))
    os.makedirs(refs, exist_ok=True)
    with open(path, 'w') as f:
        f.write(commit)


def snapshot_download(repo_id, revision='main', cache_dir=None,
                      progress=None, token=None, endpoint=ENDPOINT):
    '''Download (or reuse) the inference files; return the local path.'''
    progress = progress or Progress()
    rdir = repo_dir(repo_id, cache_dir)
    s = _session(token)
    info = repo_info(repo_id, revision, s, endpoint)
    commit = info['sha']
    snap = os.path.join(rdir, commit)
    if is_complete(snap):
        _write_ref(rdir, revision, commit)
        return snap
    os.makedirs(snap, exist_ok=True)
    names = select_files(info['files'])
    specs = {}

    def fetch(name):
        check_filename(name)
        spec = info['files'].get(name)
        if spec is None:
            raise HubError('%s is not in the repo listing' % name)
        dest = os.path.join(snap, name)
        if os.path.exists(dest):
            try:
                verify_file(dest, spec)     # cache reuse after a crash
                specs[name] = spec
                return dest
            except HubError:
                os.remove(dest)
        url = '%s/%s/resolve/%s/%s' % (endpoint, repo_id, commit, name)
        download_file(s, url, dest, spec, progress)
        specs[name] = spec
        return dest

    for name in names:
        path = fetch(name)
        if name == INDEX:
            for shard in shards_from_index(path):
                fetch(shard)
    # The weights must at least parse before the snapshot counts.
    from tinygpt.hf.safetensors import Checkpoint
    Checkpoint(snap).close()
    _write_manifest(snap, repo_id, commit, specs,
                    '%s/%s@%s' % (endpoint, repo_id, revision))
    _write_ref(rdir, revision, commit)
    return snap


# ------------------------------------------------------------------- local

def is_complete(snap):
    '''A snapshot is usable if its manifest exists and every file in it
    is present at the recorded size. (Full hashing: verify_snapshot.)'''
    man = os.path.join(snap, 'manifest.json')
    if not os.path.exists(man):
        return False
    try:
        with open(man) as f:
            files = json.load(f)['files']
        for name, spec in files.items():
            p = os.path.join(snap, check_filename(name))
            if not os.path.exists(p) or (spec.get('size') is not None and
                                         os.path.getsize(p) != spec['size']):
                return False
    except (ValueError, KeyError, HubError):
        return False
    return True


def verify_snapshot(snap):
    '''Re-hash every file against the manifest. Raises HubError.'''
    with open(os.path.join(snap, 'manifest.json')) as f:
        files = json.load(f)['files']
    for name, spec in sorted(files.items()):
        verify_file(os.path.join(snap, check_filename(name)), spec)
    return sorted(files)


def find_local(repo_id, cache_dir=None, revision='main'):
    '''The cached snapshot for repo_id, without touching the network.'''
    rdir = repo_dir(repo_id, cache_dir)
    ref = os.path.join(rdir, 'refs', revision.replace('/', '--'))
    cands = []
    if os.path.exists(ref):
        with open(ref) as f:
            cands.append(f.read().strip())
    if os.path.isdir(rdir):
        rest = [d for d in os.listdir(rdir) if d != 'refs'
                and os.path.isdir(os.path.join(rdir, d))]
        rest.sort(key=lambda d: -os.path.getmtime(os.path.join(rdir, d)))
        cands.extend(d for d in rest if d not in cands)
    for c in cands:
        snap = os.path.join(rdir, c)
        if is_complete(snap):
            return snap
    return None


def import_local(source, repo_id, cache_dir=None, progress=None):
    '''Copy a model from a local directory or .zip into the cache.

    For when the files arrived some other way (a git-lfs checkout, AirDrop,
    a zip from a desktop). The same allowlist applies, and the safetensors
    header must parse. Hashes are computed and recorded in the manifest,
    so verify_snapshot() can catch later corruption.
    '''
    progress = progress or Progress()
    rdir = repo_dir(repo_id, cache_dir)
    if zipfile.is_zipfile(source):
        zf = zipfile.ZipFile(source)
        members = {}
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            parts = zi.filename.replace('\\', '/').split('/')
            if any(p in ('', '.', '..') for p in parts[:-1]) or \
                    zi.filename.startswith('/') or len(parts) > 2:
                continue          # nested or unsafe paths: ignore
            members.setdefault(parts[-1], zi)

        def opener(name):
            return zf.open(members[name])
        sizes = {n: zi.file_size for n, zi in members.items()}
    elif os.path.isdir(source):
        sizes = {n: os.path.getsize(os.path.join(source, n))
                 for n in os.listdir(source)
                 if os.path.isfile(os.path.join(source, n))}

        def opener(name):
            return open(os.path.join(source, name), 'rb')
    else:
        raise HubError('%s is neither a directory nor a zip file' % source)

    names = select_files({n: {} for n in sizes})
    if INDEX in names:
        names.extend(n for n in sizes if SHARD_RE.match(n))
    tmp = os.path.join(rdir, 'import.tmp')
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    specs = {}
    model_hash = hashlib.sha256()
    for name in names:
        check_filename(name)
        h = hashlib.sha256()
        done = 0
        with opener(name) as src, open(os.path.join(tmp, name), 'wb') as dst:
            while True:
                b = src.read(CHUNK)
                if not b:
                    break
                dst.write(b)
                h.update(b)
                done += len(b)
                progress(name, done, sizes[name])
        progress(name, done, sizes[name], final=True)
        specs[name] = {'size': done, 'sha256': h.hexdigest()}
        model_hash.update(name.encode() + h.digest())
    if INDEX in names:
        need = set(shards_from_index(os.path.join(tmp, INDEX)))
        if not need <= set(names):
            raise HubError('index names shards that are not present: %s'
                           % ', '.join(sorted(need - set(names))))
    from tinygpt.hf.safetensors import Checkpoint
    Checkpoint(tmp).close()
    commit = 'local-' + model_hash.hexdigest()[:16]
    snap = os.path.join(rdir, commit)
    if os.path.exists(snap):
        shutil.rmtree(snap)
    os.replace(tmp, snap)
    _write_manifest(snap, repo_id, commit, specs,
                    'import:' + os.path.abspath(source))
    _write_ref(rdir, 'main', commit)
    return snap
