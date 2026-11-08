# coding: utf-8
'''
Copy every file next to this script into your tiny_gpt folder,
overwriting what is there. Nothing is deleted.

Unzip the update, then run this script from the unzipped folder.
'''

import os
import shutil

# Your tiny_gpt folder. None = ~/Documents/tiny_gpt (Pythonista's default).
TARGET = None

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {os.path.basename(__file__), '.DS_Store'}


def main():
    target = TARGET or os.path.expanduser('~/Documents/tiny_gpt')
    if not os.path.isdir(os.path.join(target, 'tinygpt')):
        print('%s does not look like tiny_gpt; set TARGET at the top of '
              'this script.' % target)
        return
    if os.path.realpath(HERE) == os.path.realpath(target):
        print('run this from the unzipped folder, not from tiny_gpt')
        return
    n = 0
    for d, dirs, files in os.walk(HERE):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        rel = os.path.relpath(d, HERE)
        for name in files:
            if name in SKIP and rel == '.':
                continue
            dst_dir = os.path.normpath(os.path.join(target, rel))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copyfile(os.path.join(d, name), os.path.join(dst_dir, name))
            print('  %s' % os.path.normpath(os.path.join(rel, name)))
            n += 1
    print('copied %d files into %s' % (n, target))


main()
