# coding: utf-8
'''
Install one native-16bit package (a ZIP) into a tiny_gpt checkout.

Standard library only; works in Pythonista and desktop Python 3.6+.

    python install_native_16bit.py --zip native_16bit_diagnostic.zip \\
        --repo /path/to/tiny_gpt --dry-run
    python install_native_16bit.py --zip ... --repo ...        # asks first
    python install_native_16bit.py --zip ... --repo ... --yes  # no prompt

Pythonista: import the package ZIP anywhere under Documents (the tiny_gpt
folder is a good place) and run this script. It finds the ZIP by its
contents -- a renamed duplicate such as "native_16bit_diagnostic 2.zip" is
fine, and so is a ZIP you already tapped to extract into a folder. It finds
the repository, shows the plan, and asks once before changing anything.
Nothing has to be deleted or overwritten by hand: that is this script's job.

The installer copies itself into tiny_gpt/, so for later stages only the
new ZIP needs importing; run tiny_gpt/install_native_16bit.py again. If a
package carries a newer installer, it updates that copy too. Afterwards it
offers to move the used ZIP (and stray installer copies) into
tiny_gpt/.native16/imported/, so the next import never collides with an
old file name.

What it guarantees:
  * Only the paths listed in the ZIP's manifest.json are touched. A file is
    never replaced or removed merely because the ZIP lacks it.
  * The manifest and every ZIP entry are validated first: no absolute
    paths, no '..', no paths into .git, hashes must match. Nothing is
    extracted with extractall(); each listed file is read by name.
  * Every file that will be replaced or deleted is first copied into
    <repo>/.native16/backups/<timestamp>_<version>/. Deletions are moves
    into that folder, not real deletes.
  * A file that exists but is not the version this package expects (i.e.
    you edited it, or created a file of the same name) is a conflict and is
    left alone unless you pass --force (it is still backed up).
  * If anything fails halfway, the changes already made are undone from
    the backup, and any file that could not be restored is named.
  * Running it again is harmless: files already at the packaged version are
    skipped and no new backup is made.
  * <repo>/.native16/state.json records what each package installed, which
    is what rollback_native_16bit.py uses to undo it.
'''

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import posixpath
import py_compile
import shutil
import sys
import tempfile
import traceback
import zipfile

INSTALLER_VERSION = 2
# Present in the installer and rollback scripts: an existing file containing
# it is an older copy of one of them, which a package may update in place.
TOOL_MARKER = 'native16-packaging-tool'
MANIFEST_FORMAT = 1
STAGES = ('diagnostic', 'proof-of-concept', 'benchmark', 'optimization',
          'integration')
STATE_DIR = '.native16'
# Files whose presence identifies a tiny_gpt checkout with the HF path.
REPO_MARKERS = ('tinygpt/backend.py', 'tinygpt/hf/llama.py',
                'tinygpt/hf/safetensors.py', 'hf_run.py')
REPO_MARKER_TEXT = ('tinygpt/backend.py', 'cblas_sgemv')
MAX_FILE_BYTES = 16 << 20


class PackageError(Exception):
    pass


# ----------------------------------------------------------------- helpers

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _read_file(path):
    with open(path, 'rb') as f:
        return f.read()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_rel_path(p):
    '''A repository-relative POSIX path that cannot escape the repo.'''
    if not isinstance(p, str) or not p:
        raise PackageError('path must be a non-empty string: %r' % (p,))
    if '\\' in p or '\0' in p or ':' in p:
        raise PackageError('path has a forbidden character: %r' % p)
    if p.startswith('/') or posixpath.normpath(p) != p:
        raise PackageError('path is not a normalized relative path: %r' % p)
    parts = p.split('/')
    if any(x in ('', '.', '..') for x in parts):
        raise PackageError('path has an empty, "." or ".." part: %r' % p)
    if parts[0] in ('.git', STATE_DIR):
        raise PackageError('path points into %s: %r' % (parts[0], p))
    if len(p) > 240:
        raise PackageError('path too long: %r' % p)
    return p


def target_path(repo, rel):
    '''Absolute path for rel inside repo, refusing symlinked components.'''
    full = os.path.join(repo, *rel.split('/'))
    real_repo = os.path.realpath(repo)
    cur = repo
    for part in rel.split('/'):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            raise PackageError('%s: %s is a symlink; refusing to write '
                               'through it' % (rel, cur))
    real = os.path.realpath(full)
    if not (real == real_repo or real.startswith(real_repo + os.sep)):
        raise PackageError('%s resolves outside the repository' % rel)
    return full


def ask(prompt, default=False):
    try:
        ans = input(prompt)
    except EOFError:
        return default
    ans = ans.strip().lower()
    if not ans:
        return default
    return ans.startswith('y')


def now_stamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


# ---------------------------------------------------------------- manifest

def validate_manifest(m):
    if not isinstance(m, dict):
        raise PackageError('manifest is not a JSON object')
    for key in ('version', 'stage', 'files'):
        if key not in m:
            raise PackageError('manifest is missing %r' % key)
    if m.get('format', 1) != MANIFEST_FORMAT:
        raise PackageError('manifest format %r not supported by this '
                           'installer (wants %d); get a newer installer'
                           % (m.get('format'), MANIFEST_FORMAT))
    if m.get('min_installer', 1) > INSTALLER_VERSION:
        raise PackageError('package needs installer version %r or newer'
                           % m['min_installer'])
    if not isinstance(m['version'], str) or not m['version']:
        raise PackageError('manifest version must be a non-empty string')
    if m['stage'] not in STAGES:
        raise PackageError('unknown stage %r' % m['stage'])
    files = m['files']
    if not isinstance(files, dict) or set(files) - {'add', 'replace',
                                                    'delete'}:
        raise PackageError('manifest files must have only add/replace/delete')
    seen = set()
    for kind in ('add', 'replace', 'delete'):
        lst = files.get(kind, [])
        if not isinstance(lst, list):
            raise PackageError('files.%s must be a list' % kind)
        for p in lst:
            check_rel_path(p)
            if p in seen:
                raise PackageError('%s listed more than once' % p)
            seen.add(p)
    sha = m.get('sha256', {})
    for p in files.get('add', []) + files.get('replace', []):
        if not isinstance(sha.get(p), str) or len(sha[p]) != 64:
            raise PackageError('manifest has no sha256 for %s' % p)
    orig = m.get('original_sha256', {})
    for p in files.get('replace', []) + files.get('delete', []):
        if p not in orig:
            raise PackageError('replace/delete of %s needs original_sha256 '
                               '(list of acceptable current hashes)' % p)
        if not isinstance(orig[p], list):
            raise PackageError('original_sha256[%s] must be a list' % p)
    reasons = m.get('delete_reasons', {})
    for p in files.get('delete', []):
        if not reasons.get(p):
            raise PackageError('deleting %s needs a delete_reasons entry' % p)
    deps = m.get('depends_on', [])
    if not isinstance(deps, list) or not all(isinstance(d, str)
                                             for d in deps):
        raise PackageError('depends_on must be a list of version strings')
    return m


def _dir_names(root):
    '''Files under an extracted package folder, as relative POSIX names.'''
    out = []
    for d, dirs, files in os.walk(root):
        for n in dirs + files:
            if os.path.islink(os.path.join(d, n)):
                raise PackageError('package folder contains a symlink: %s'
                                   % os.path.join(d, n))
        dirs[:] = [x for x in dirs if x != '__pycache__']
        rel = os.path.relpath(d, root)
        for n in files:
            if n == '.DS_Store' or n.endswith('.pyc'):
                continue
            out.append(n if rel == '.' else
                       '/'.join(rel.split(os.sep) + [n]))
    return out


class Package(object):
    '''A package ZIP, or the folder you get by extracting one.'''

    def __init__(self, path):
        self.path = path
        if os.path.isdir(path):
            names = _dir_names(path)
            self._read = lambda n: _read_file(os.path.join(
                path, *n.split('/')))
            self._size = lambda n: os.path.getsize(os.path.join(
                path, *n.split('/')))
        else:
            try:
                zf = zipfile.ZipFile(path)
            except (OSError, zipfile.BadZipfile) as e:
                raise PackageError('cannot open %s: %s' % (path, e))
            names = zf.namelist()
            self._read = zf.read
            self._size = lambda n: zf.getinfo(n).file_size
        for n in names:
            # Every member must be safe, even ones we never extract.
            if n.endswith('/'):
                check_rel_path(n.rstrip('/'))
            else:
                check_rel_path(n)
        if 'manifest.json' not in names:
            raise PackageError('%s has no manifest.json' % path)
        try:
            self.manifest = validate_manifest(
                json.loads(self._read('manifest.json').decode('utf-8')))
        except ValueError as e:
            raise PackageError('manifest.json is not valid JSON: %s' % e)
        m = self.manifest
        payload = set(n[len('files/'):] for n in names
                      if n.startswith('files/') and not n.endswith('/'))
        wanted = set(m['files'].get('add', []) + m['files'].get('replace',
                                                                []))
        missing = wanted - payload
        if missing:
            raise PackageError('ZIP lacks listed files: %s'
                               % ', '.join(sorted(missing)))
        extra = payload - wanted
        if extra:
            raise PackageError('ZIP has files not in the manifest: %s'
                               % ', '.join(sorted(extra)))
        self.data = {}
        for p in wanted:
            if self._size('files/' + p) > MAX_FILE_BYTES:
                raise PackageError('%s is implausibly large' % p)
            b = self._read('files/' + p)
            if sha256_bytes(b) != m['sha256'][p]:
                raise PackageError('%s does not match its manifest hash' % p)
            self.data[p] = b

    @property
    def version(self):
        return self.manifest['version']


# -------------------------------------------------------------------- repo

def verify_repo(repo):
    problems = []
    for rel in REPO_MARKERS:
        if not os.path.isfile(os.path.join(repo, *rel.split('/'))):
            problems.append('missing %s' % rel)
    if not problems:
        try:
            with open(os.path.join(repo, *REPO_MARKER_TEXT[0].split('/')),
                      'rb') as f:
                if REPO_MARKER_TEXT[1].encode() not in f.read():
                    problems.append('%s lacks %r' % REPO_MARKER_TEXT)
        except OSError as e:
            problems.append(str(e))
    return problems


def load_state(repo):
    p = os.path.join(repo, STATE_DIR, 'state.json')
    if not os.path.exists(p):
        return {'packages': []}
    with open(p) as f:
        return json.load(f)


def save_state(repo, state):
    d = os.path.join(repo, STATE_DIR)
    _ensure_state_dir(repo)
    tmp = os.path.join(d, 'state.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(d, 'state.json'))


def _ensure_state_dir(repo):
    d = os.path.join(repo, STATE_DIR)
    if not os.path.isdir(d):
        os.makedirs(d)
    gi = os.path.join(d, '.gitignore')
    if not os.path.exists(gi):
        with open(gi, 'w') as f:
            f.write('# installer backups and state; not part of the repo\n*\n')


def active_packages(state):
    return [p for p in state['packages'] if p.get('status') == 'installed']


def tracked_hashes(state, rel):
    '''Hashes a native16 package itself wrote to rel (still active).'''
    out = set()
    for p in active_packages(state):
        f = p['files'].get(rel)
        if f and f.get('sha_after'):
            out.add(f['sha_after'])
    return out


# -------------------------------------------------------------------- plan

class Step(object):
    __slots__ = ('rel', 'kind', 'action', 'note', 'cur_sha', 'new_sha')

    def __init__(self, rel, kind, action, note='', cur_sha=None,
                 new_sha=None):
        self.rel, self.kind, self.action, self.note = rel, kind, action, note
        self.cur_sha, self.new_sha = cur_sha, new_sha


def make_plan(pkg, repo, state, force):
    '''One Step per manifest path. action is one of
    add | replace | delete | skip | conflict.'''
    m = pkg.manifest
    steps = []
    for kind in ('add', 'replace', 'delete'):
        for rel in m['files'].get(kind, []):
            full = target_path(repo, rel)
            exists = os.path.isfile(full)
            if os.path.exists(full) and not exists:
                steps.append(Step(rel, kind, 'conflict',
                                  'exists and is not a regular file'))
                continue
            cur = sha256_file(full) if exists else None
            new = m['sha256'].get(rel)
            ours = tracked_hashes(state, rel)
            allowed = set(m.get('original_sha256', {}).get(rel, [])) | ours
            if kind == 'add':
                if not exists:
                    s = Step(rel, kind, 'add')
                elif cur == new:
                    s = Step(rel, kind, 'skip', 'already this version')
                elif cur in ours:
                    s = Step(rel, kind, 'replace',
                             'update of a file an earlier native16 package '
                             'installed')
                elif TOOL_MARKER.encode() in _read_file(full):
                    s = Step(rel, kind, 'replace',
                             'update of an older installer/rollback copy')
                else:
                    s = Step(rel, kind, 'conflict',
                             'a different file already exists here')
            elif kind == 'replace':
                if not exists:
                    s = Step(rel, kind, 'conflict', 'file to replace is '
                             'missing')
                elif cur == new:
                    s = Step(rel, kind, 'skip', 'already this version')
                elif cur in allowed:
                    s = Step(rel, kind, 'replace')
                else:
                    s = Step(rel, kind, 'conflict',
                             'file differs from the version this package '
                             'was built against (local edits?)')
            else:
                if not exists:
                    s = Step(rel, kind, 'skip', 'already absent')
                elif cur in allowed:
                    s = Step(rel, kind, 'delete',
                             m['delete_reasons'][rel])
                else:
                    s = Step(rel, kind, 'conflict',
                             'file differs from the version this package '
                             'expects to delete')
            s.cur_sha, s.new_sha = cur, new
            if s.action == 'conflict' and force and s.note != \
                    'exists and is not a regular file':
                s.note += '  [--force: will back up and proceed]'
                s.action = ('delete' if kind == 'delete' else
                            'add' if not exists else 'replace')
            steps.append(s)
    return steps


def print_plan(pkg, repo, steps, out=print):
    m = pkg.manifest
    out('package   %s  (stage: %s)' % (m['version'], m['stage']))
    out('zip       %s' % pkg.path)
    out('repo      %s' % repo)
    if m.get('depends_on'):
        out('requires  %s' % ', '.join(m['depends_on']))
    if m.get('requires_device_testing'):
        out('device    this package must be tested on the iOS device before '
            'the next stage')
    out('')
    for s in steps:
        out('  %-9s %-48s %s' % (s.action.upper(), s.rel, s.note))
    counts = {}
    for s in steps:
        counts[s.action] = counts.get(s.action, 0) + 1
    out('')
    out('  ' + ', '.join('%d %s' % (v, k) for k, v in sorted(counts.items())))


# ------------------------------------------------------------------- apply

def _write_atomic(full, data):
    d = os.path.dirname(full)
    if not os.path.isdir(d):
        os.makedirs(d)
    fd, tmp = tempfile.mkstemp(prefix='.native16_', dir=d)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, full)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def apply_plan(pkg, repo, steps, log=print):
    '''Perform the plan. Returns the state record. On failure, undoes what
    was done and raises PackageError naming anything not restored.'''
    todo = [s for s in steps if s.action in ('add', 'replace', 'delete')]
    if any(s.action == 'conflict' for s in steps):
        raise PackageError('plan has conflicts; nothing changed')
    stamp = now_stamp()
    _ensure_state_dir(repo)
    bdir = os.path.join(repo, STATE_DIR, 'backups',
                        '%s_%s' % (stamp, pkg.version))
    n = 1
    while os.path.exists(bdir):
        n += 1
        bdir = os.path.join(repo, STATE_DIR, 'backups', '%s_%s_%d'
                            % (stamp, pkg.version, n))
    os.makedirs(os.path.join(bdir, 'files'))
    record = {'version': pkg.version, 'stage': pkg.manifest['stage'],
              'installed_at': datetime.datetime.now().isoformat(),
              'zip': os.path.abspath(pkg.path),
              'zip_sha256': (sha256_file(pkg.path)
                             if os.path.isfile(pkg.path) else None),
              'backup_dir': os.path.relpath(bdir, repo),
              'status': 'in-progress', 'files': {}}
    logpath = os.path.join(bdir, 'install_log.json')

    def save_log():
        with open(logpath, 'w') as f:
            json.dump(record, f, indent=1, sort_keys=True)

    save_log()
    done = []      # (step, backup path or None)
    try:
        for s in todo:
            full = target_path(repo, s.rel)
            backup = None
            if os.path.exists(full):
                backup = os.path.join(bdir, 'files', *s.rel.split('/'))
                if not os.path.isdir(os.path.dirname(backup)):
                    os.makedirs(os.path.dirname(backup))
                # Recorded before the move/copy so that a failure from here
                # on is undone from whatever reached the backup.
                done.append((s, backup))
                if s.action == 'delete':
                    shutil.move(full, backup)
                else:
                    shutil.copy2(full, backup)
                if sha256_file(backup) != s.cur_sha:
                    raise PackageError('backup of %s does not match the '
                                       'original' % s.rel)
            else:
                done.append((s, None))
            if s.action != 'delete':
                _write_atomic(full, pkg.data[s.rel])
                if sha256_file(full) != s.new_sha:
                    raise PackageError('%s did not write correctly' % s.rel)
            record['files'][s.rel] = {
                'action': s.action, 'sha_before': s.cur_sha,
                'sha_after': None if s.action == 'delete' else s.new_sha,
                'backup': (os.path.relpath(backup, repo) if backup
                           else None)}
            log('  %-8s %s' % (s.action, s.rel))
            save_log()
    except BaseException as e:
        log('')
        log('FAILED: %s' % (e,))
        log('restoring the files changed so far ...')
        failed = undo(repo, done, log)
        record['status'] = 'failed-restored' if not failed else \
            'failed-partially-restored'
        record['error'] = repr(e)
        record['not_restored'] = failed
        save_log()
        if failed:
            raise PackageError('install failed and these files could NOT be '
                               'restored (originals are in %s): %s'
                               % (bdir, ', '.join(failed)))
        raise PackageError('install failed; every change was undone (%s)'
                           % (e,))
    record['status'] = 'installed'
    save_log()
    with open(os.path.join(bdir, 'install_log.txt'), 'w') as f:
        f.write('%s installed %s\n' % (record['installed_at'], pkg.version))
        for rel, r in sorted(record['files'].items()):
            f.write('  %-8s %s\n' % (r['action'], rel))
    return record


def undo(repo, done, log=print):
    failed = []
    for s, backup in reversed(done):
        full = os.path.join(repo, *s.rel.split('/'))
        try:
            if backup is not None and not os.path.exists(backup):
                # Failed before the backup was made: original untouched.
                log('  unchanged %s' % s.rel)
                continue
            if backup is not None:
                if os.path.exists(full):
                    os.remove(full)
                shutil.copy2(backup, full)
            elif os.path.exists(full):
                os.remove(full)
            log('  restored %s' % s.rel)
        except Exception as e:
            log('  could not restore %s: %s' % (s.rel, e))
            failed.append(s.rel)
    return failed


# -------------------------------------------------------------- validation

def validate_install(pkg, repo, run_tests, log=print):
    '''Byte-compile installed Python files; optionally run listed tests.'''
    ok = True
    m = pkg.manifest
    for rel in m['files'].get('add', []) + m['files'].get('replace', []):
        if rel.endswith('.py'):
            try:
                py_compile.compile(os.path.join(repo, *rel.split('/')),
                                   cfile=os.path.join(tempfile.gettempdir(),
                                                      'native16_check.pyc'),
                                   doraise=True)
            except py_compile.PyCompileError as e:
                log('  compile error in %s: %s' % (rel, e))
                ok = False
    if ok:
        log('  all installed Python files compile')
    tests = m.get('tests', [])
    if run_tests and tests and ok:
        import unittest
        log('  running %s' % ', '.join(tests))
        sys.path.insert(0, repo)
        old = os.getcwd()
        try:
            os.chdir(repo)
            for name in list(sys.modules):
                if name == 'tinygpt' or name.startswith(('tinygpt.',
                                                         'tests.')):
                    del sys.modules[name]
            suite = unittest.defaultTestLoader.loadTestsFromNames(tests)
            res = unittest.TextTestRunner(verbosity=1).run(suite)
            ok = res.wasSuccessful()
        finally:
            os.chdir(old)
            sys.path.remove(repo)
    elif tests and not run_tests:
        log('  tests not run (pass --run-tests, or run: python -m unittest '
            '%s)' % ' '.join(tests))
    return ok


# --------------------------------------------------------------------- run

def check_dependencies(pkg, state):
    have = set(p['version'] for p in active_packages(state))
    return [d for d in pkg.manifest.get('depends_on', []) if d not in have]


def prepare(pkg_path, repo, force=False):
    '''Validate everything and compute the plan; changes nothing.'''
    repo = os.path.abspath(repo)
    problems = verify_repo(repo)
    if problems:
        raise PackageError('%s does not look like a tiny_gpt checkout: %s'
                           % (repo, '; '.join(problems)))
    pkg = Package(pkg_path)
    state = load_state(repo)
    missing = check_dependencies(pkg, state)
    if missing:
        raise PackageError('%s requires %s to be installed first'
                           % (pkg.version, ', '.join(missing)))
    return repo, pkg, make_plan(pkg, repo, state, force)


def execute(pkg, repo, steps, run_tests, log=print):
    log('')
    log('installing ...')
    record = apply_plan(pkg, repo, steps, log)
    log('')
    log('validating ...')
    ok = validate_install(pkg, repo, run_tests, log)
    state = load_state(repo)
    state['packages'].append(record)
    save_state(repo, state)
    log('')
    log('backup and log: %s' % os.path.join(repo, record['backup_dir']))
    log('to undo:        run rollback_native_16bit.py (in %s)' % repo)
    if not ok:
        log('')
        log('INSTALLED, BUT VALIDATION FAILED (see above). Run the rollback '
            'script if this is unexpected.')
        return 2
    log('')
    log('SUCCESS: %s installed.' % pkg.version)
    return 0


def install(pkg_path, repo, dry_run=False, force=False, yes=False,
            run_tests=False, log=print, confirm=None):
    repo, pkg, steps = prepare(pkg_path, repo, force)
    print_plan(pkg, repo, steps, log)
    if any(s.action == 'conflict' for s in steps):
        log('')
        log('CONFLICTS: these files exist but are not what the package '
            'expects. Nothing will be changed. Inspect them, then re-run '
            'with --force to back them up and overwrite.')
        return 3
    if not [s for s in steps if s.action != 'skip']:
        log('')
        log('Nothing to do: %s is already installed.' % pkg.version)
        return 0
    if dry_run:
        log('')
        log('DRY RUN: nothing was changed.')
        return 0
    if not yes:
        confirm = confirm or (lambda: ask('\nApply these changes? [y/N] '))
        if not confirm():
            log('Cancelled; nothing was changed.')
            return 1
    rc = execute(pkg, repo, steps, run_tests, log)
    if rc == 0 and pkg.manifest.get('next_steps'):
        log('')
        log(pkg.manifest['next_steps'])
    return rc


# --------------------------------------------------------------- discovery

SKIP_DIRS = ('.git', STATE_DIR, 'hf_models', 'site-packages', '__pycache__',
             'native16_results', '.Trash')
MAX_ZIP_BYTES = 64 << 20


def peek_manifest(path):
    '''The manifest of a native16 package (ZIP or folder), else None.
    Never raises: anything unreadable simply is not a package.'''
    try:
        if os.path.isdir(path):
            mp = os.path.join(path, 'manifest.json')
            if not os.path.isfile(mp):
                return None
            m = json.loads(_read_file(mp).decode('utf-8'))
        else:
            if os.path.getsize(path) > MAX_ZIP_BYTES:
                return None
            with zipfile.ZipFile(path) as zf:
                m = json.loads(zf.read('manifest.json').decode('utf-8'))
        if isinstance(m, dict) and str(m.get('version', '')).startswith(
                'native-16bit'):
            return m
    except Exception:
        pass
    return None


def discover(roots, max_depth=3):
    '''Every native16 package under roots, newest first within a version.

    Packages are recognised by content, not name, so Pythonista's renamed
    duplicates ("x 2.zip") and extracted folders are found too.
    '''
    found, seen = [], set()
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        base = root.count(os.sep)
        for d, dirs, files in os.walk(root):
            real = os.path.realpath(d)
            if real in seen:
                dirs[:] = []
                continue
            seen.add(real)
            m = peek_manifest(d)
            if m is not None:
                found.append({'path': d, 'manifest': m,
                              'mtime': os.path.getmtime(d)})
                dirs[:] = []
                continue
            if d.count(os.sep) - base >= max_depth:
                dirs[:] = []
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS
                             and not os.path.islink(os.path.join(d, x)))
            for n in sorted(files):
                if not n.lower().endswith('.zip'):
                    continue
                p = os.path.join(d, n)
                if os.path.realpath(p) in seen:
                    continue
                seen.add(os.path.realpath(p))
                m = peek_manifest(p)
                if m is not None:
                    found.append({'path': p, 'manifest': m,
                                  'mtime': os.path.getmtime(p)})
    found.sort(key=lambda f: (-STAGES.index(f['manifest']['stage'])
                              if f['manifest'].get('stage') in STAGES else 0,
                              f['manifest']['version'], -f['mtime']))
    return found


def search_roots(here, repo=None):
    docs = os.path.expanduser('~/Documents')
    roots = [here, os.path.dirname(here)]
    if repo:
        roots += [repo, os.path.dirname(repo)]
    roots += [os.path.join(docs, 'Inbox'), docs]
    out = []
    for r in roots:
        r = os.path.abspath(r)
        if r not in out:
            out.append(r)
    return out


def stray_tool_copies(roots, repo):
    '''Loose copies of these scripts (e.g. "install_native_16bit 2.py")
    outside the repository's own copies, in the top level of each root.'''
    keep = set(os.path.realpath(os.path.join(repo, n)) for n in
               ('install_native_16bit.py', 'rollback_native_16bit.py'))
    out = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for n in sorted(os.listdir(r)):
            p = os.path.join(r, n)
            low = n.lower()
            if (low.startswith(('install_native_16bit', 'rollback_native_16bit'))
                    and low.endswith('.py') and os.path.isfile(p)
                    and os.path.realpath(p) not in keep
                    and TOOL_MARKER.encode() in _read_file(p)
                    and p not in out):
                out.append(p)
    return out


def tidy(repo, paths, log=print):
    '''Move used packages and stray script copies into
    <repo>/.native16/imported/<stamp>/. Moves, never deletes.'''
    if not paths:
        return []
    _ensure_state_dir(repo)
    dest = os.path.join(repo, STATE_DIR, 'imported', now_stamp())
    os.makedirs(dest)
    moved = []
    for p in paths:
        if not os.path.exists(p):
            continue
        target = os.path.join(dest, os.path.basename(p.rstrip(os.sep)))
        k = 1
        while os.path.exists(target):
            k += 1
            target = os.path.join(dest, '%d_%s' % (k, os.path.basename(p)))
        try:
            shutil.move(p, target)
            moved.append(p)
            log('  moved %s' % p)
        except Exception as e:
            log('  could not move %s: %s' % (p, e))
    log('  -> %s' % dest)
    return moved


def tidy_candidates(found, used_version, roots, repo):
    '''Packages at or below the version just handled (the one used and any
    duplicates or older stages), plus stray script copies.'''
    order = {v: i for i, v in enumerate(STAGES)}
    used_stage = None
    for f in found:
        if f['manifest']['version'] == used_version:
            used_stage = order.get(f['manifest'].get('stage'), 0)
    out = []
    state = load_state(repo)
    installed = set(p['version'] for p in state['packages'])
    for f in found:
        v = f['manifest']['version']
        st = order.get(f['manifest'].get('stage'), 0)
        if v == used_version or v in installed or (
                used_stage is not None and st < used_stage):
            out.append(f['path'])
    # Don't move a folder that contains the repository itself.
    real_repo = os.path.realpath(repo)
    out = [p for p in out if not real_repo.startswith(
        os.path.realpath(p) + os.sep)]
    return out + [p for p in stray_tool_copies(roots, repo)
                  if not any(p.startswith(q + os.sep) for q in out)]


# ------------------------------------------------------------- interactive

def find_repo_candidates(here):
    docs = os.path.expanduser('~/Documents')
    cands = [here, os.path.dirname(here),
             os.path.dirname(os.path.dirname(here)),
             os.path.join(here, 'tiny_gpt'),
             os.path.join(os.path.dirname(here), 'tiny_gpt'),
             os.path.join(docs, 'tiny_gpt')]
    if os.path.isdir(docs):
        for n in sorted(os.listdir(docs)):
            cands.append(os.path.join(docs, n))
    seen, out = set(), []
    for c in cands:
        c = os.path.abspath(c)
        if c not in seen and os.path.isdir(c) and not verify_repo(c):
            seen.add(c)
            out.append(c)
    return out


def choose(items, what, labels=None, default=0):
    labels = labels or items
    for i, x in enumerate(labels):
        print('  %d) %s%s' % (i + 1, x, '   <- default' if i == default
                                       else ''))
    try:
        ans = input('Choose %s [1-%d, Enter = %d]: '
                    % (what, len(items), default + 1)).strip()
    except EOFError:
        ans = ''
    if not ans:
        return items[default]
    try:
        k = int(ans)
    except ValueError:
        return None
    return items[k - 1] if 1 <= k <= len(items) else None


HOW_TO_IMPORT = '''
No native-16bit package found.

To get one into Pythonista:
  1. Save the package ZIP (e.g. native_16bit_diagnostic.zip) to the Files
     app.
  2. In Pythonista's file browser, open the tiny_gpt folder, tap "+" (or
     "..."), choose Import, then Files, and pick the ZIP.
     Any folder under Documents works; renamed duplicates are fine.
  3. Run this installer again.
'''


def interactive(here=None):
    here = here or os.path.dirname(os.path.abspath(__file__))
    print('native-16bit installer (v%d)' % INSTALLER_VERSION)
    print('')
    repos = find_repo_candidates(here)
    if len(repos) == 1:
        repo = repos[0]
    elif repos:
        repo = choose(repos, 'the tiny_gpt folder')
    else:
        repo = input('Could not find tiny_gpt. Path to the tiny_gpt '
                     'folder: ').strip()
    if not repo:
        print('Cancelled.')
        return 1
    print('repository: %s' % repo)
    roots = search_roots(here, repo)
    found = discover(roots)
    if not found:
        print(HOW_TO_IMPORT)
        return 1
    state = load_state(repo) if not verify_repo(repo) else {'packages': []}
    installed = set(p['version'] for p in active_packages(state))
    # One entry per version: the newest copy. Others are duplicates.
    by_version = []
    for f in found:
        if f['manifest']['version'] not in [g['manifest']['version']
                                            for g in by_version]:
            by_version.append(f)
    labels = ['%s%s   (%s)' % (f['manifest']['version'],
                               '  [installed]' if f['manifest']['version']
                               in installed else '', f['path'])
              for f in by_version]
    pending = [i for i, f in enumerate(by_version)
               if f['manifest']['version'] not in installed]
    default = pending[0] if pending else 0
    if len(by_version) == 1:
        pick = by_version[0]
        print('package:    %s' % labels[0])
    else:
        print('packages found:')
        pick = choose(by_version, 'a package', labels, default)
        if pick is None:
            print('Cancelled.')
            return 1
    print('')
    repo, pkg, steps = prepare(pick['path'], repo)
    print_plan(pkg, repo, steps)
    conflicts = [s for s in steps if s.action == 'conflict']
    todo = [s for s in steps if s.action != 'skip']
    rc = 0
    if conflicts:
        print('')
        print('Some files already exist and differ from what this package '
              'expects')
        print('(usually: an older copy you copied in by hand, or your own '
              'edits).')
        print('If you continue, each is first saved to tiny_gpt/.native16/'
              'backups/ and')
        print('then replaced. The rollback script can put them back.')
        if not ask('\nReplace them? [y/N] '):
            print('Cancelled; nothing was changed.')
            return 3
        repo, pkg, steps = prepare(pick['path'], repo, force=True)
        todo = [s for s in steps if s.action != 'skip']
    if not todo:
        print('')
        print('Nothing to do: %s is already installed.' % pkg.version)
    else:
        if not conflicts and not ask('\nInstall? [Y/n] ', True):
            print('Cancelled; nothing was changed.')
            return 1
        rc = execute(pkg, repo, steps, run_tests=True)
    if rc in (0, 2):
        cands = tidy_candidates(found, pkg.version, roots, repo)
        if cands:
            print('')
            print('Tidy up: move these used / duplicate files out of the way,')
            print('into tiny_gpt/.native16/imported/ (nothing is deleted):')
            for c in cands:
                print('  %s' % c)
            if ask('Move them? [Y/n] ', True):
                tidy(repo, cands)
    if rc == 0 and pkg.manifest.get('next_steps'):
        print('')
        print(pkg.manifest['next_steps'])
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Install a native-16bit package into tiny_gpt.')
    ap.add_argument('--zip', required=True,
                    help='package ZIP (or the folder extracted from it)')
    ap.add_argument('--repo', required=True, help='tiny_gpt checkout')
    ap.add_argument('--dry-run', action='store_true',
                    help='show the plan; change nothing')
    ap.add_argument('--force', action='store_true',
                    help='back up and overwrite conflicting files')
    ap.add_argument('--yes', action='store_true', help='do not prompt')
    ap.add_argument('--run-tests', action='store_true',
                    help="run the manifest's tests after installing")
    ap.add_argument('--tidy', action='store_true',
                    help='afterwards, move the used ZIP into '
                         '<repo>/.native16/imported/')
    a = ap.parse_args(argv)
    rc = install(a.zip, a.repo, a.dry_run, a.force, a.yes, a.run_tests)
    if rc == 0 and a.tidy and not a.dry_run:
        tidy(os.path.abspath(a.repo), [a.zip])
    return rc


def _entry():
    try:
        return interactive() if len(sys.argv) == 1 else main()
    except PackageError as e:
        print('')
        print('ERROR: %s' % e)
        return 4
    except Exception:
        traceback.print_exc()
        print('')
        print('ERROR: unexpected failure (details above).')
        return 5


if __name__ == '__main__':
    rc = _entry()
    if rc and len(sys.argv) > 1:
        sys.exit(rc)
