# coding: utf-8
'''packaging/native16 installer, rollback and builder, on a scratch copy of
the repository markers. Host-side only; not shipped in the packages.'''

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = os.path.join(REPO, 'packaging', 'native16')
sys.path.insert(0, PKG)
import build_packages as bp            # noqa: E402
import install_native_16bit as inst    # noqa: E402
import rollback_native_16bit as rb     # noqa: E402
sys.path.remove(PKG)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def quiet(*a):
    pass


def slurp(path):
    with open(path, 'rb') as f:
        return f.read()


def make_zip(path, manifest, files, extra=None):
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('manifest.json', json.dumps(manifest))
        for p, b in files.items():
            zf.writestr('files/' + p, b)
        for p, b in (extra or {}).items():
            zf.writestr(p, b)
    return path


def manifest(version='t-1', add=(), replace=(), delete=(), data=None,
             orig=None, reasons=None, deps=()):
    data = data or {}
    return {'format': 1, 'version': version, 'stage': 'diagnostic',
            'depends_on': list(deps),
            'files': {'add': list(add), 'replace': list(replace),
                      'delete': list(delete)},
            'sha256': {p: sha(data[p]) for p in list(add) + list(replace)},
            'original_sha256': orig or {}, 'delete_reasons': reasons or {}}


class Base(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, 'tiny_gpt')
        for rel in inst.REPO_MARKERS + ('README.md',):
            src = os.path.join(REPO, *rel.split('/'))
            dst = os.path.join(self.repo, *rel.split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def install(self, z, **kw):
        kw.setdefault('yes', True)
        return inst.install(z, self.repo, log=quiet, **kw)

    def rollback(self, **kw):
        kw.setdefault('yes', True)
        return rb.rollback(self.repo, log=quiet, **kw)

    def p(self, rel):
        return os.path.join(self.repo, *rel.split('/'))

    def read(self, rel):
        with open(self.p(rel), 'rb') as f:
            return f.read()

    def backups(self):
        d = os.path.join(self.repo, '.native16', 'backups')
        return sorted(os.listdir(d)) if os.path.isdir(d) else []


class RealPackageTest(Base):
    '''The diagnostic ZIP exactly as build_packages.py makes it.'''

    def setUp(self):
        Base.setUp(self)
        self.zip = bp.build('diagnostic', self.tmp)
        self.spec = bp.STAGES['diagnostic']

    def test_reproducible(self):
        a = slurp(self.zip)
        d2 = os.path.join(self.tmp, 'again')
        os.makedirs(d2)
        b = slurp(bp.build('diagnostic', d2))
        self.assertEqual(a, b)

    def test_dry_run_changes_nothing(self):
        self.assertEqual(self.install(self.zip, dry_run=True), 0)
        for rel in self.spec['add']:
            self.assertFalse(os.path.exists(self.p(rel)))
        self.assertFalse(os.path.exists(os.path.join(self.repo,
                                                     '.native16')))

    def test_install_rerun_rollback(self):
        self.assertEqual(self.install(self.zip, run_tests=False), 0)
        for rel in self.spec['add']:
            self.assertEqual(self.read(rel), bp._read(rel))
        self.assertEqual(len(self.backups()), 1)
        # idempotent: second run does nothing, makes no backup
        self.assertEqual(self.install(self.zip), 0)
        self.assertEqual(len(self.backups()), 1)
        st = inst.load_state(self.repo)
        self.assertEqual([p['status'] for p in st['packages']],
                         ['installed'])
        self.assertEqual(self.rollback(dry_run=True), 0)
        self.assertTrue(os.path.exists(self.p(self.spec['add'][0])))
        self.assertEqual(self.rollback(), 0)
        for rel in self.spec['add']:
            self.assertFalse(os.path.exists(self.p(rel)))
        # untouched production files
        self.assertEqual(self.read('tinygpt/backend.py'),
                         slurp(os.path.join(REPO, 'tinygpt', 'backend.py')))
        st = rb.load_state(self.repo)
        self.assertEqual(st['packages'][0]['status'], 'rolled-back')
        # and it can be installed again afterwards
        self.assertEqual(self.install(self.zip), 0)

    def test_user_file_is_a_conflict(self):
        rel = self.spec['add'][0]
        os.makedirs(os.path.dirname(self.p(rel)), exist_ok=True)
        with open(self.p(rel), 'wb') as f:
            f.write(b'my own script\n')
        self.assertEqual(self.install(self.zip), 3)
        self.assertEqual(self.read(rel), b'my own script\n')
        self.assertFalse(os.path.exists(self.p(self.spec['add'][1])))
        self.assertEqual(self.install(self.zip, force=True), 0)
        self.assertNotEqual(self.read(rel), b'my own script\n')
        # rollback brings the user's file back
        self.assertEqual(self.rollback(), 0)
        self.assertEqual(self.read(rel), b'my own script\n')

    def test_installed_tests_pass(self):
        self.assertEqual(self.install(self.zip, run_tests=True), 0)

    def test_not_a_repo(self):
        with self.assertRaises(inst.PackageError):
            inst.install(self.zip, self.tmp, log=quiet, yes=True)

    def test_failure_midway_restores(self):
        rel0 = self.spec['add'][0]
        os.makedirs(os.path.dirname(self.p(rel0)), exist_ok=True)
        with open(self.p(rel0), 'wb') as f:
            f.write(b'old\n')
        calls = []
        real = inst._write_atomic

        def boom(full, data):
            calls.append(full)
            if len(calls) == 3:
                raise OSError('disk full (simulated)')
            real(full, data)
        inst._write_atomic = boom
        try:
            with self.assertRaises(inst.PackageError) as cm:
                self.install(self.zip, force=True)
        finally:
            inst._write_atomic = real
        self.assertIn('every change was undone', str(cm.exception))
        self.assertEqual(self.read(rel0), b'old\n')
        for rel in self.spec['add'][1:]:
            self.assertFalse(os.path.exists(self.p(rel)), rel)
        self.assertEqual(inst.load_state(self.repo)['packages'], [])


class PythonistaFlowTest(Base):
    '''Discovery, extracted folders, self-update and tidy-up, laid out the
    way Pythonista's Documents folder looks.'''

    def setUp(self):
        Base.setUp(self)
        self.zip = bp.build('diagnostic', self.tmp)
        self.version = bp.STAGES['diagnostic']['version']

    def test_discovers_renamed_zip_and_extracted_folder(self):
        dup = os.path.join(self.repo, 'native_16bit_diagnostic 2.zip')
        shutil.copy2(self.zip, dup)
        ext = os.path.join(self.tmp, 'native_16bit_diagnostic')
        with zipfile.ZipFile(self.zip) as zf:
            zf.extractall(ext)
        with open(os.path.join(self.tmp, 'notes.zip'), 'wb') as f:
            f.write(b'not a zip')
        found = inst.discover([self.tmp])
        paths = sorted(f['path'] for f in found)
        self.assertEqual(paths, sorted([self.zip, dup, ext]))

    def test_install_from_extracted_folder(self):
        ext = os.path.join(self.tmp, 'native_16bit_diagnostic')
        with zipfile.ZipFile(self.zip) as zf:
            zf.extractall(ext)
        self.assertEqual(self.install(ext), 0)
        self.assertEqual(self.read('install_native_16bit.py'),
                         bp._read('install_native_16bit.py'))

    def test_older_installer_copy_is_updated_not_conflict(self):
        with open(self.p('install_native_16bit.py'), 'wb') as f:
            f.write(b"# old copy\nTOOL_MARKER = 'native16-packaging-tool'\n")
        self.assertEqual(self.install(self.zip), 0)
        self.assertEqual(self.read('install_native_16bit.py'),
                         bp._read('install_native_16bit.py'))

    def test_unrelated_file_with_tool_name_is_conflict(self):
        with open(self.p('install_native_16bit.py'), 'wb') as f:
            f.write(b'print("something of mine")\n')
        self.assertEqual(self.install(self.zip), 3)

    def test_interactive_end_to_end(self):
        # ~/Documents/tiny_gpt is the repo; the user imported the ZIP twice
        # (Pythonista renamed the second) and a loose installer copy.
        home = os.path.join(self.tmp, 'home')
        docs = os.path.join(home, 'Documents')
        os.makedirs(docs)
        repo = os.path.join(docs, 'tiny_gpt')
        shutil.move(self.repo, repo)
        self.repo = repo
        z1 = os.path.join(docs, 'native_16bit_diagnostic.zip')
        z2 = os.path.join(repo, 'native_16bit_diagnostic 2.zip')
        shutil.copy2(self.zip, z1)
        shutil.copy2(self.zip, z2)
        stray = os.path.join(docs, 'install_native_16bit 2.py')
        with open(stray, 'wb') as f:
            f.write(bp._read('install_native_16bit.py'))
        answers = iter(['y', 'y'])      # Install? / Move them?
        old_home = os.environ.get('HOME')
        os.environ['HOME'] = home
        import builtins
        real_input = builtins.input
        builtins.input = lambda prompt='': next(answers)
        real_print = builtins.print
        builtins.print = quiet
        try:
            rc = inst.interactive(here=docs)
        finally:
            builtins.input = real_input
            builtins.print = real_print
            if old_home is None:
                del os.environ['HOME']
            else:
                os.environ['HOME'] = old_home
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(self.p('tools/native16_diagnostics.py')))
        for p in (z1, z2, stray):
            self.assertFalse(os.path.exists(p), p)
        imported = os.path.join(repo, '.native16', 'imported')
        moved = []
        for d, _, files in os.walk(imported):
            moved += files
        self.assertEqual(sorted(moved),
                         sorted([os.path.basename(z1), os.path.basename(z2),
                                 os.path.basename(stray)]))
        # The repo's own copies of the tools stay put.
        self.assertTrue(os.path.exists(self.p('install_native_16bit.py')))
        self.assertTrue(os.path.exists(self.p('rollback_native_16bit.py')))


class ManifestSafetyTest(Base):

    def z(self, m, files, extra=None):
        return make_zip(os.path.join(self.tmp, 'p.zip'), m, files, extra)

    def assertRefused(self, z, needle):
        with self.assertRaises(inst.PackageError) as cm:
            self.install(z)
        self.assertIn(needle, str(cm.exception))

    def test_traversal_in_manifest(self):
        for bad in ('../evil.py', '/etc/x', 'a/../../b', '.git/config',
                    'a\\b', 'C:/x', 'a//b', './a'):
            d = {bad: b'x'}
            m = manifest(add=[bad], data=d)
            with zipfile.ZipFile(os.path.join(self.tmp, 'p.zip'), 'w') as zf:
                zf.writestr('manifest.json', json.dumps(m))
            with self.assertRaises(inst.PackageError):
                self.install(os.path.join(self.tmp, 'p.zip'))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, 'evil.py')))

    def test_traversal_in_unlisted_member(self):
        d = {'tools/a.py': b'x'}
        z = self.z(manifest(add=['tools/a.py'], data=d), d,
                   extra={'../../escape.txt': b'x'})
        self.assertRefused(z, '".." part')

    def test_hash_mismatch(self):
        m = manifest(add=['tools/a.py'], data={'tools/a.py': b'x'})
        self.assertRefused(self.z(m, {'tools/a.py': b'y'}),
                           'does not match its manifest hash')

    def test_unlisted_payload(self):
        d = {'tools/a.py': b'x'}
        z = self.z(manifest(add=['tools/a.py'], data=d),
                   dict(d, **{'tools/b.py': b'y'}))
        self.assertRefused(z, 'not in the manifest')

    def test_missing_payload(self):
        d = {'tools/a.py': b'x'}
        self.assertRefused(self.z(manifest(add=['tools/a.py'], data=d), {}),
                           'lacks listed files')

    def test_delete_needs_reason_and_hash(self):
        m = manifest(delete=['README.md'],
                     orig={'README.md': [sha(self.read('README.md'))]})
        self.assertRefused(self.z(m, {}), 'delete_reasons')
        m = manifest(delete=['README.md'], reasons={'README.md': 'why'})
        self.assertRefused(self.z(m, {}), 'original_sha256')

    def test_dependency(self):
        d = {'tools/a.py': b'x'}
        z = self.z(manifest(add=['tools/a.py'], data=d, deps=['t-0']), d)
        self.assertRefused(z, 'requires t-0')

    def test_symlink_target_refused(self):
        os.symlink(self.tmp, self.p('tools'))
        d = {'tools/a.py': b'x'}
        self.assertRefused(self.z(manifest(add=['tools/a.py'], data=d), d),
                           'symlink')

    def test_replace_and_delete_roundtrip(self):
        orig_readme = self.read('README.md')
        orig_run = self.read('hf_run.py')
        new = b'# replaced\n'
        m = manifest(version='t-2', replace=['README.md'],
                     delete=['hf_run.py'], data={'README.md': new},
                     orig={'README.md': [sha(orig_readme)],
                           'hf_run.py': [sha(orig_run)]},
                     reasons={'hf_run.py': 'test deletion'})
        # hf_run.py is a repo marker; deleting it would break detection of
        # the repo for the rollback's own sanity, which rollback does not
        # need, so this exercises the general mechanism.
        z = self.z(m, {'README.md': new})
        self.assertEqual(self.install(z), 0)
        self.assertEqual(self.read('README.md'), new)
        self.assertFalse(os.path.exists(self.p('hf_run.py')))
        self.assertEqual(self.rollback(), 0)
        self.assertEqual(self.read('README.md'), orig_readme)
        self.assertEqual(self.read('hf_run.py'), orig_run)

    def test_replace_of_locally_edited_file_is_conflict(self):
        new = b'# replaced\n'
        m = manifest(replace=['README.md'], data={'README.md': new},
                     orig={'README.md': ['0' * 64]})
        self.assertEqual(self.install(self.z(m, {'README.md': new})), 3)
        self.assertNotEqual(self.read('README.md'), new)

    def test_rollback_conflict_when_edited_after_install(self):
        d = {'tools/a.py': b'x = 1\n'}
        self.assertEqual(self.install(self.z(manifest(add=['tools/a.py'],
                                                      data=d), d)), 0)
        with open(self.p('tools/a.py'), 'wb') as f:
            f.write(b'x = 2  # my edit\n')
        self.assertEqual(self.rollback(), 3)
        self.assertEqual(self.read('tools/a.py'), b'x = 2  # my edit\n')
        self.assertEqual(self.rollback(force=True), 0)
        self.assertFalse(os.path.exists(self.p('tools/a.py')))
        saved = rb.load_state(self.repo)['packages'][0]['rollback_saved_dir']
        with open(os.path.join(self.repo, saved, 'tools', 'a.py'),
                  'rb') as f:
            self.assertEqual(f.read(), b'x = 2  # my edit\n')

    def test_later_package_updates_earlier_file(self):
        d1 = {'tools/a.py': b'v1\n'}
        self.assertEqual(self.install(self.z(manifest('t-1', add=['tools/a.py'],
                                                      data=d1), d1)), 0)
        d2 = {'tools/a.py': b'v2\n'}
        z2 = make_zip(os.path.join(self.tmp, 'p2.zip'),
                      manifest('t-2', add=['tools/a.py'], data=d2,
                               deps=['t-1']), d2)
        self.assertEqual(self.install(z2), 0)
        self.assertEqual(self.read('tools/a.py'), b'v2\n')
        with self.assertRaises(rb.RollbackError):
            self.rollback(version='t-1')
        self.assertEqual(self.rollback(), 0)            # undoes t-2
        self.assertEqual(self.read('tools/a.py'), b'v1\n')
        self.assertEqual(self.rollback(), 0)            # undoes t-1
        self.assertFalse(os.path.exists(self.p('tools/a.py')))


if __name__ == '__main__':
    unittest.main()
