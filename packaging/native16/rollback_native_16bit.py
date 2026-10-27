# coding: utf-8
'''
Undo a native-16bit package installed by install_native_16bit.py.

Standard library only; works in Pythonista and desktop Python 3.6+.

    python rollback_native_16bit.py --repo /path/to/tiny_gpt --list
    python rollback_native_16bit.py --repo /path/to/tiny_gpt --dry-run
    python rollback_native_16bit.py --repo /path/to/tiny_gpt          # latest
    python rollback_native_16bit.py --repo ... --version native-16bit-diagnostic-1

Pythonista: press Run; it finds the repository and asks what to undo.

It reads <repo>/.native16/state.json, written at install time, and for each
file that package touched:
  added    -> moved out of the repo into .native16/rollbacks/<stamp>/
              (never permanently deleted)
  replaced -> original restored from the install backup
  deleted  -> original restored from the install backup
A file you have changed since the install is a conflict and is left alone
unless --force (your changed copy is still saved under .native16/rollbacks).
Only the most recently installed package can be undone unless --force, so
later packages are undone first.
'''

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import traceback

STATE_DIR = '.native16'
# Lets the installer recognise an older copy of this script and update it.
TOOL_MARKER = 'native16-packaging-tool'


class RollbackError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ask(prompt, default=False):
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return default
    return default if not ans else ans.startswith('y')


def safe_join(repo, rel):
    parts = rel.split('/')
    if (rel.startswith('/') or '\\' in rel
            or any(p in ('', '.', '..') for p in parts)):
        raise RollbackError('unsafe path in state file: %r' % rel)
    full = os.path.join(repo, *parts)
    real_repo = os.path.realpath(repo)
    if not os.path.realpath(full).startswith(real_repo + os.sep):
        raise RollbackError('%s resolves outside the repository' % rel)
    return full


def load_state(repo):
    p = os.path.join(repo, STATE_DIR, 'state.json')
    if not os.path.exists(p):
        raise RollbackError('no %s: nothing was installed by the native16 '
                            'installer here' % p)
    with open(p) as f:
        return json.load(f)


def save_state(repo, state):
    p = os.path.join(repo, STATE_DIR, 'state.json')
    with open(p + '.tmp', 'w') as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(p + '.tmp', p)


def active(state):
    return [p for p in state['packages'] if p.get('status') == 'installed']


def plan(repo, rec, force):
    steps = []
    for rel, f in sorted(rec['files'].items()):
        full = safe_join(repo, rel)
        exists = os.path.isfile(full)
        cur = sha256_file(full) if exists else None
        backup = (os.path.join(repo, *f['backup'].split('/'))
                  if f.get('backup') else None)
        if f['action'] == 'delete':
            if exists:
                a, note = 'conflict', 'a file has reappeared here'
            elif backup and os.path.isfile(backup):
                a, note = 'restore', 'bring back the deleted original'
            else:
                a, note = 'conflict', 'backup copy missing'
        else:
            if not exists:
                a = 'restore' if f.get('sha_before') else 'skip'
                note = ('original restored from backup' if a == 'restore'
                        else 'already absent')
            elif cur != f['sha_after']:
                a, note = 'conflict', 'changed since install'
            elif f.get('sha_before'):
                a, note = 'restore', 'original restored from backup'
            else:
                a, note = 'remove', 'added by the package'
            if a == 'restore' and not (backup and os.path.isfile(backup)):
                a, note = 'conflict', 'backup copy missing'
        if a == 'conflict' and force and note == 'changed since install':
            a = 'restore' if f.get('sha_before') else 'remove'
            note += '  [--force: your copy is saved first]'
        steps.append((rel, a, note, full, backup, exists))
    return steps


def rollback(repo, version=None, dry_run=False, force=False, yes=False,
             log=print, confirm=None):
    repo = os.path.abspath(repo)
    state = load_state(repo)
    act = active(state)
    if not act:
        log('No installed native16 packages to undo.')
        return 0
    if version is None:
        rec = act[-1]
    else:
        m = [p for p in act if p['version'] == version]
        if not m:
            raise RollbackError('%s is not installed (installed: %s)'
                                % (version, ', '.join(p['version']
                                                      for p in act)))
        rec = m[-1]
        if rec is not act[-1] and not force:
            raise RollbackError('%s was installed after %s; undo it first '
                                '(or --force)' % (act[-1]['version'],
                                                  version))
    steps = plan(repo, rec, force)
    log('undo      %s  (installed %s)' % (rec['version'],
                                          rec['installed_at']))
    log('repo      %s' % repo)
    log('')
    for rel, a, note, _, _, _ in steps:
        log('  %-9s %-48s %s' % (a.upper(), rel, note))
    if any(a == 'conflict' for _, a, _, _, _, _ in steps):
        log('')
        log('CONFLICTS: nothing changed. Inspect the files above; --force '
            'saves your copies under .native16/rollbacks/ and proceeds.')
        return 3
    if dry_run:
        log('')
        log('DRY RUN: nothing was changed.')
        return 0
    if not yes:
        confirm = confirm or (lambda: ask('\nUndo these changes? [y/N] '))
        if not confirm():
            log('Cancelled; nothing was changed.')
            return 1
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    trash = os.path.join(repo, STATE_DIR, 'rollbacks',
                         '%s_%s' % (stamp, rec['version']))
    failed = []
    for rel, a, note, full, backup, exists in steps:
        try:
            if a == 'skip':
                continue
            if exists:
                keep = os.path.join(trash, *rel.split('/'))
                if not os.path.isdir(os.path.dirname(keep)):
                    os.makedirs(os.path.dirname(keep))
                shutil.move(full, keep)
            if a == 'restore':
                d = os.path.dirname(full)
                if not os.path.isdir(d):
                    os.makedirs(d)
                shutil.copy2(backup, full)
            log('  %-8s %s' % (a, rel))
        except Exception as e:
            log('  FAILED   %s: %s' % (rel, e))
            failed.append(rel)
    rec['status'] = 'rolled-back' if not failed else 'rollback-incomplete'
    rec['rolled_back_at'] = datetime.datetime.now().isoformat()
    rec['rollback_saved_dir'] = os.path.relpath(trash, repo)
    if failed:
        rec['rollback_failed'] = failed
    save_state(repo, state)
    log('')
    if failed:
        log('ROLLBACK INCOMPLETE for: %s. Originals are in %s.'
            % (', '.join(failed), os.path.join(repo, rec['backup_dir'])))
        return 2
    log('SUCCESS: %s undone. Removed files were moved to %s'
        % (rec['version'], trash))
    return 0


def list_packages(repo, log=print):
    state = load_state(os.path.abspath(repo))
    for p in state['packages']:
        log('  %-34s %-20s %s' % (p['version'], p.get('status'),
                                  p.get('installed_at')))
    return 0


def find_repo():
    here = os.path.dirname(os.path.abspath(__file__))
    docs = os.path.expanduser('~/Documents')
    cands = [here, os.path.dirname(here), os.path.join(here, 'tiny_gpt'),
             os.path.join(os.path.dirname(here), 'tiny_gpt'),
             os.path.join(docs, 'tiny_gpt')]
    if os.path.isdir(docs):
        cands += [os.path.join(docs, n) for n in sorted(os.listdir(docs))]
    for c in cands:
        if os.path.isfile(os.path.join(c, STATE_DIR, 'state.json')):
            return os.path.abspath(c)
    return None


def interactive():
    repo = find_repo()
    if repo is None or not ask('Repository %s? [Y/n] ' % repo, True):
        repo = input('Path to the tiny_gpt folder: ').strip()
    list_packages(repo)
    print('')
    rc = rollback(repo, dry_run=True)
    if rc == 0 and ask('\nUndo the most recent package now? [y/N] '):
        rc = rollback(repo, yes=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description='Undo a native16 package.')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--version', help='package version (default: latest)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--yes', action='store_true')
    a = ap.parse_args(argv)
    if a.list:
        return list_packages(a.repo)
    return rollback(a.repo, a.version, a.dry_run, a.force, a.yes)


def _entry():
    try:
        return interactive() if len(sys.argv) == 1 else main()
    except RollbackError as e:
        print('\nERROR: %s' % e)
        return 4
    except Exception:
        traceback.print_exc()
        print('\nERROR: unexpected failure (details above).')
        return 5


if __name__ == '__main__':
    rc = _entry()
    if rc and len(sys.argv) > 1:
        sys.exit(rc)
