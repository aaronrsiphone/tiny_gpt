# coding: utf-8
'''The downloader, end to end, against a fake Hub on localhost.

The fake server speaks the two endpoints the downloader uses (the
revision API and /resolve/), honours Range requests, and can be told to
drop a connection halfway or serve corrupted bytes, so resume and
validation are exercised without the real network.
'''

import base64
import hashlib
import json
import os
import shutil
import tempfile
import threading
import unittest
import zipfile

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
except ImportError:              # pragma: no cover
    BaseHTTPRequestHandler = HTTPServer = None

from tests.hf_common import fixture
from tinygpt.hf import hub

try:
    import requests  # noqa: F401
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

COMMIT = 'a' * 40


def git_sha1(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


class FakeHub(object):

    def __init__(self, files, lfs=('model.safetensors',)):
        self.files = dict(files)          # name -> bytes
        self.lfs = set(lfs)
        self.requests = []
        self.cut_after = {}               # name -> bytes before a drop
        self.corrupt = set()
        hub_ = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hub_.requests.append((self.path, self.headers.get('Range')))
                if self.path.startswith('/api/models/'):
                    return self.api()
                if '/resolve/' in self.path:
                    return self.resolve()
                self.send_error(404)

            def api(self):
                sib = []
                for n, b in hub_.files.items():
                    e = {'rfilename': n, 'size': len(b)}
                    if n in hub_.lfs:
                        e['lfs'] = {'sha256': hashlib.sha256(b).hexdigest(),
                                    'size': len(b)}
                    else:
                        e['blobId'] = git_sha1(b)
                    sib.append(e)
                body = json.dumps({'sha': COMMIT, 'siblings': sib}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def resolve(self):
                name = self.path.split('/resolve/')[1].split('/', 1)[1]
                if name not in hub_.files:
                    return self.send_error(404)
                b = hub_.files[name]
                if name in hub_.corrupt:
                    b = b[:-1] + bytes([b[-1] ^ 1])
                start = 0
                rng = self.headers.get('Range')
                if rng:
                    start = int(rng.split('=')[1].split('-')[0])
                    self.send_response(206)
                    self.send_header('Content-Range', 'bytes %d-%d/%d'
                                     % (start, len(b) - 1, len(b)))
                else:
                    self.send_response(200)
                self.send_header('Content-Length', str(len(b) - start))
                self.end_headers()
                cut = hub_.cut_after.pop(name, None)
                if cut is not None:
                    self.wfile.write(b[start:start + cut])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(b[start:])

        self.server = HTTPServer(('127.0.0.1', 0), H)
        self.url = 'http://127.0.0.1:%d' % self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def model_files():
    d = fixture('tiny_llama_tied_bf16.json')
    return {
        'config.json': json.dumps(d['config']).encode(),
        'tokenizer.json': json.dumps(
            fixture('bpe_tiny.json')['tokenizer']).encode(),
        'generation_config.json': b'{"eos_token_id": 2}',
        'model.safetensors': base64.b64decode(d['safetensors']),
        'README.md': b'not downloaded',
        'training_args.bin': b'\x80\x04pickle',
        'onnx/model.onnx': b'nope',
    }


def quiet(*a, **k):
    pass


@unittest.skipUnless(HAVE_REQUESTS and HTTPServer, 'needs requests')
class SnapshotTest(unittest.TestCase):

    def setUp(self):
        self.files = model_files()
        self.fake = FakeHub(self.files)
        self.cache = tempfile.mkdtemp()
        # Talk to the local server directly, never via a proxy.
        self._env = {k: os.environ.pop(k) for k in list(os.environ)
                     if k.lower() in ('http_proxy', 'https_proxy',
                                      'all_proxy')}

    def tearDown(self):
        self.fake.close()
        os.environ.update(self._env)
        shutil.rmtree(self.cache, ignore_errors=True)

    def dl(self):
        return hub.snapshot_download('org/tiny', cache_dir=self.cache,
                                     progress=quiet, endpoint=self.fake.url)

    def test_download_resume_reuse_offline(self):
        self.fake.cut_after['model.safetensors'] = 1000
        saved, hub.STREAM_CHUNK = hub.STREAM_CHUNK, 100
        try:
            snap = self.dl()
        finally:
            hub.STREAM_CHUNK = saved
        got = sorted(os.listdir(snap))
        self.assertEqual(got, ['config.json', 'generation_config.json',
                               'manifest.json', 'model.safetensors',
                               'tokenizer.json'])
        with open(os.path.join(snap, 'model.safetensors'), 'rb') as f:
            self.assertEqual(f.read(), self.files['model.safetensors'])
        ranges = [r for p, r in self.fake.requests
                  if p.endswith('model.safetensors') and r]
        self.assertEqual(ranges, ['bytes=1000-'])        # it resumed

        n = len(self.fake.requests)
        self.assertEqual(self.dl(), snap)                 # cache reuse
        self.assertEqual(len(self.fake.requests), n + 1)  # metadata only

        self.fake.close()                                 # offline now
        self.assertEqual(hub.find_local('org/tiny', self.cache), snap)
        hub.verify_snapshot(snap)
        from tinygpt.hf.llama import LlamaModel
        LlamaModel.from_dir(snap, ctx_limit=16)
        self.fake = FakeHub({})                           # for tearDown

    def test_corrupt_download_rejected(self):
        self.fake.corrupt.add('model.safetensors')
        with self.assertRaises(hub.HubError):
            self.dl()
        self.assertIsNone(hub.find_local('org/tiny', self.cache))
        snapdir = os.path.join(self.cache, 'org--tiny', COMMIT)
        self.assertFalse(any(n.startswith('model.safetensors')
                             for n in os.listdir(snapdir)))

    def test_damaged_cache_detected(self):
        snap = self.dl()
        with open(os.path.join(snap, 'model.safetensors'), 'r+b') as f:
            f.seek(100)
            f.write(b'\xff')
        with self.assertRaises(hub.HubError):
            hub.verify_snapshot(snap)
        with open(os.path.join(snap, 'model.safetensors'), 'ab') as f:
            f.write(b'x')
        self.assertIsNone(hub.find_local('org/tiny', self.cache))

    def test_pickle_only_repo_refused(self):
        self.fake.close()
        f = dict(self.files)
        del f['model.safetensors']
        f['pytorch_model.bin'] = b'\x80\x04'
        self.fake = FakeHub(f, lfs=('pytorch_model.bin',))
        with self.assertRaises(hub.HubError) as cm:
            self.dl()
        self.assertIn('pickle', str(cm.exception))


class NamesTest(unittest.TestCase):

    def test_filenames(self):
        for ok in ('config.json', 'model.safetensors',
                   'model-00001-of-00003.safetensors'):
            hub.check_filename(ok)
        for bad in ('../config.json', '/etc/passwd', 'a/config.json',
                    'pytorch_model.bin', 'modeling_llama.py', '.hidden',
                    'model-1-of-2.safetensors', 'x\\config.json', ''):
            with self.assertRaises(hub.HubError):
                hub.check_filename(bad)

    def test_repo_ids(self):
        hub.check_repo_id('HuggingFaceTB/SmolLM2-135M-Instruct')
        for bad in ('noslash', '../x/y', 'a/b/c', 'a/..', '/a/b', 'a b/c'):
            with self.assertRaises(hub.HubError):
                hub.check_repo_id(bad)

    def test_select(self):
        files = {n: {} for n in model_files()}
        self.assertEqual(hub.select_files(files),
                         ['config.json', 'tokenizer.json',
                          'generation_config.json', 'model.safetensors'])
        del files['tokenizer.json']
        with self.assertRaises(hub.HubError):
            hub.select_files(files)


class ImportTest(unittest.TestCase):

    def test_zip_import(self):
        td = tempfile.mkdtemp()
        try:
            z = os.path.join(td, 'm.zip')
            with zipfile.ZipFile(z, 'w') as zf:
                for n, b in model_files().items():
                    if '/' not in n:
                        zf.writestr('m/' + n, b)
                zf.writestr('../evil.json', b'{}')
                zf.writestr('m/sub/config.json', b'{"bad": 1}')
            cache = os.path.join(td, 'cache')
            snap = hub.import_local(z, 'org/tiny', cache, progress=quiet)
            self.assertEqual(hub.find_local('org/tiny', cache), snap)
            self.assertNotIn('training_args.bin', os.listdir(snap))
            self.assertFalse(os.path.exists(os.path.join(td, 'evil.json')))
            with open(os.path.join(snap, 'config.json'), 'rb') as f:
                self.assertNotIn(b'bad', f.read())
            hub.verify_snapshot(snap)
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
