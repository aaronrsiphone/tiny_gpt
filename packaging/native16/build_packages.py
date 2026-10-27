# coding: utf-8
'''
Build the native-16bit package ZIPs from this checkout, reproducibly.

    python packaging/native16/build_packages.py [stage ...] [--out dist]

Same sources -> byte-identical ZIPs: entries are sorted, timestamps fixed
at 1980-01-01, permissions fixed, compression level fixed. Each ZIP holds

    manifest.json            validated by install_native_16bit.py
    README.md                the stage's instructions
    CHANGELOG.md
    install_native_16bit.py  convenience copies; not installed
    rollback_native_16bit.py
    files/<repo path>        the payload, exactly the manifest's add/replace

and dist/SHA256SUMS lists every artifact.
'''

import argparse
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# One entry per stage. Later stages are added here only after the device
# results of the previous one have been reviewed.
STAGES = {
    'diagnostic': {
        'version': 'native-16bit-diagnostic-1',
        'stage': 'diagnostic',
        'zip': 'native_16bit_diagnostic.zip',
        'readme': 'docs/native16_diagnostic.md',
        'depends_on': [],
        'add': ['install_native_16bit.py',
                'rollback_native_16bit.py',
                'tools/native16_diagnostics.py',
                'tests/test_native16_diagnostics.py',
                'tests/native16_fake_bnns.c',
                'docs/native16_diagnostic.md',
                'docs/native16_feasibility.md'],
        'replace': [],
        'delete': [],
        'tests': ['tests.test_native16_diagnostics'],
        'next_steps': (
            'NEXT: open tools/native16_diagnostics.py and press Run; answer '
            'y to the probe question. Send back the .txt and .json files '
            'from native16_results/. Do not install another stage until '
            'the results have been reviewed.'),
    },
}

FIXED_TIME = (1980, 1, 1, 0, 0, 0)

# Payload paths whose source lives elsewhere in this checkout. The two
# scripts are installed at the repo root so later stages need only a ZIP.
SOURCES = {
    'install_native_16bit.py': 'packaging/native16/install_native_16bit.py',
    'rollback_native_16bit.py': 'packaging/native16/rollback_native_16bit.py',
}


def _read(rel):
    rel = SOURCES.get(rel, rel)
    with open(os.path.join(REPO, *rel.split('/')), 'rb') as f:
        return f.read()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def manifest_for(spec):
    files = spec['add'] + spec['replace']
    m = {
        'format': 1,
        'min_installer': 1,
        'version': spec['version'],
        'stage': spec['stage'],
        'depends_on': spec['depends_on'],
        'files': {'add': spec['add'], 'replace': spec['replace'],
                  'delete': spec['delete']},
        'sha256': {p: _sha(_read(p)) for p in files},
        # Hashes of the production files a replace/delete was built against.
        'original_sha256': spec.get('original_sha256', {}),
        'delete_reasons': spec.get('delete_reasons', {}),
        'backup_required': True,
        'requires_device_testing': True,
        'tests': spec.get('tests', []),
        'next_steps': spec.get('next_steps', ''),
    }
    return m


def _add(zf, name, data):
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    zf.writestr(info, data, compresslevel=9)


def build(stage, out_dir):
    spec = STAGES[stage]
    m = manifest_for(spec)
    entries = {
        'manifest.json': (json.dumps(m, indent=1, sort_keys=True)
                          + '\n').encode('utf-8'),
        'README.md': _read(spec['readme']),
        'CHANGELOG.md': _read('packaging/native16/CHANGELOG.md'),
        'install_native_16bit.py':
            _read('packaging/native16/install_native_16bit.py'),
        'rollback_native_16bit.py':
            _read('packaging/native16/rollback_native_16bit.py'),
    }
    for p in spec['add'] + spec['replace']:
        entries['files/' + p] = _read(p)
    path = os.path.join(out_dir, spec['zip'])
    with zipfile.ZipFile(path, 'w') as zf:
        for name in sorted(entries):
            _add(zf, name, entries[name])
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stages', nargs='*', default=sorted(STAGES))
    ap.add_argument('--out', default=os.path.join(REPO, 'dist'))
    a = ap.parse_args(argv)
    if not os.path.isdir(a.out):
        os.makedirs(a.out)
    arts = []
    for s in a.stages:
        arts.append(build(s, a.out))
    for n in ('install_native_16bit.py', 'rollback_native_16bit.py'):
        dst = os.path.join(a.out, n)
        with open(dst, 'wb') as f:
            f.write(_read('packaging/native16/' + n))
        arts.append(dst)
    lines = []
    for p in sorted(os.listdir(a.out)):
        full = os.path.join(a.out, p)
        if p != 'SHA256SUMS' and os.path.isfile(full):
            with open(full, 'rb') as f:
                lines.append('%s  %s' % (_sha(f.read()), p))
    with open(os.path.join(a.out, 'SHA256SUMS'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    for p in arts:
        print(p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
