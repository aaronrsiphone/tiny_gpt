# coding: utf-8
'''
Build a chat corpus by downloading dataset files directly. No API.

hf_corpus.py goes through HF's dataset-viewer, which paginates at 100 rows,
rate-limits per IP, and returns 502 when it is unhappy. This fetches whole
files from their permanent URLs instead: one request each, resumable, cached
on disk.

    python direct_corpus.py list
    python direct_corpus.py dolly
    python direct_corpus.py alpaca-cleaned dolly --out big.txt
    python direct_corpus.py all --max-out 600 --out huge.txt

Combining sources is the point. A 5 MB corpus at 8192 tokens/step and 16000
steps is 26 epochs, and the train/val gap widens accordingly. Every source
added cuts the repetition proportionally.

Output is ASCII-folded identically to alpaca_corpus.py and hf_corpus.py, so
a model trained on any of them has a vocabulary of exactly printable ASCII
plus newline, and every corpus is interchangeable for eval.

NOTE: only the alpaca-cleaned URL has been verified reachable.
The dolly URL came from the HF file listing and is believed correct but was
not testable from where this was written. If one 404s, the others still run:
failures are reported and skipped, not fatal.
'''

import json
import os
import sys
import urllib.error
import urllib.request

# (url, cache filename, format, description)
SOURCES = {
    'alpaca-cleaned': (
        'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/'
        'main/alpaca_data_cleaned.json',
        'alpaca_data_cleaned.json', 'json',
        '52K, community-corrected fork of alpaca. CC BY-NC 4.0.'),
    'dolly': (
        'https://huggingface.co/datasets/databricks/databricks-dolly-15k/'
        'resolve/main/databricks-dolly-15k.jsonl',
        'databricks-dolly-15k.jsonl', 'jsonl',
        '15K human-written. CC BY-SA 3.0. Different register from alpaca.'),
}

ALL = ['alpaca-cleaned', 'dolly']  # 'all' expands to this

# Field-name candidates, checked in order.
USER_KEYS = ['instruction', 'question', 'prompt', 'query']
ASSIST_KEYS = ['output', 'response', 'answer', 'completion']
CONTEXT_KEYS = ['input', 'context']

JUNK = ('<nooutput>', '<noinput>', '<no output>', '<no input>',
        'noinput', 'nooutput', 'N/A')

_FOLD = (('\u2019', "'"), ('\u2018', "'"), ('\u201c', '"'),
         ('\u201d', '"'), ('\u2013', '-'), ('\u2014', '-'),
         ('\u2026', '...'), ('\u00a0', ' '))


def to_ascii(s):
    for a, b in _FOLD:
        s = s.replace(a, b)
    return ''.join(c for c in s if 32 <= ord(c) < 127 or c == '\n')


def clean(s):
    return str(s or '').replace('\r\n', '\n').replace('\r', '\n').strip()


def is_junk(s):
    t = s.strip().lower()
    return (not t) or any(t == j.lower() or t.startswith(j.lower())
                          for j in JUNK)


def download(url, path):
    '''Fetch once, cache. Resumes a partial file with a Range request.'''
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        print('  cached  %s (%.1f MB)' % (path, os.path.getsize(path) / 1e6))
        return True

    part = path + '.part'
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = {'User-Agent': 'tiny-gpt-corpus/1.0'}
    if have:
        headers['Range'] = 'bytes=%d-' % have
        print('  resuming at %.1f MB' % (have / 1e6))

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as r, \
                open(part, 'ab' if have else 'wb') as f:
            total = int(r.headers.get('Content-Length') or 0) + have
            done = have
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = ('%3.0f%%' % (100.0 * done / total)) if total else ''
                sys.stderr.write('\r  %5.1f MB %s' % (done / 1e6, pct))
                sys.stderr.flush()
        sys.stderr.write('\n')
    except urllib.error.HTTPError as e:
        sys.stderr.write('\n')
        print('  HTTP %s: %s' % (e.code, url))
        return False
    except Exception as e:
        sys.stderr.write('\n')
        print('  failed: %r' % (e,))
        print('  (partial file kept at %s; rerun to resume)' % part)
        return False

    os.rename(part, path)
    return True


def read_rows(path, fmt):
    '''Yield dicts from a .json array or a .jsonl file.

    jsonl is read line by line rather than loaded whole: a malformed line
    then costs one row instead of the entire file.
    '''
    if fmt == 'json':
        with open(path, 'r', errors='replace') as f:
            for r in json.load(f):
                yield r
        return
    bad = 0
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                bad += 1
    if bad:
        print('  skipped %d malformed lines' % bad)


def pick_keys(row):
    low = {k.lower(): k for k in row}
    u = next((low[k] for k in USER_KEYS if k in low), None)
    a = next((low[k] for k in ASSIST_KEYS if k in low), None)
    c = next((low[k] for k in CONTEXT_KEYS if k in low and low[k] != u), None)
    return u, a, c


def harvest(name, max_in, max_out, mode, ascii_only):
    url, cache, fmt, _ = SOURCES[name]
    print('%s:' % name)
    if not download(url, cache):
        return [], 0, 0

    rows = list(read_rows(cache, fmt))
    if not rows:
        print('  no rows parsed')
        return [], 0, 0
    uk, ak, ck = pick_keys(rows[0])
    if not uk or not ak:
        print('  cannot identify fields in %s' % sorted(rows[0]))
        return [], 0, 0
    print('  %d rows, fields user=%s assistant=%s context=%s'
          % (len(rows), uk, ak, ck))

    out, skip_len, skip_empty = [], 0, 0
    for r in rows:
        user = clean(r.get(uk))
        asst = clean(r.get(ak))
        ctx = clean(r.get(ck)) if ck else ''
        if not user or not asst or is_junk(asst):
            skip_empty += 1
            continue
        if ascii_only:
            user, asst, ctx = to_ascii(user), to_ascii(asst), to_ascii(ctx)
        if ctx:
            user = user + '\n' + ctx
        if mode == 'truncate':
            user, asst = user[:max_in], asst[:max_out]
        elif len(user) > max_in or len(asst) > max_out:
            skip_len += 1
            continue
        out.append('U: %s\nA: %s' % (user, asst))
    print('  kept %d, skipped %d long, %d empty'
          % (len(out), skip_len, skip_empty))
    return out, skip_len, skip_empty


def build(names, out_path, max_in=180, max_out=280, mode='filter',
          ascii_only=True, shuffle=True, seed=0):
    chunks = []
    for n in names:
        got, _, _ = harvest(n, max_in, max_out, mode, ascii_only)
        chunks.extend(got)

    if not chunks:
        raise SystemExit('nothing downloaded; every source failed')

    if shuffle:
        # Interleave sources rather than leaving them in blocks, so the
        # validation tail is not drawn from whichever dataset happened to
        # land last. With contiguous blocks the held-out split measures one
        # source only and stops being comparable to the training mix.
        import random
        random.Random(seed).shuffle(chunks)

    text = '\n\n'.join(chunks) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)

    vocab = sorted(set(text))
    lens = sorted(len(c) for c in chunks)
    model_vocab = set(chr(c) for c in range(32, 127)) | {'\n'}
    print('\ntotal    %d exchanges from %s' % (len(chunks), ', '.join(names)))
    print('chars    %d (%.1f MB)' % (len(text), len(text) / 1e6))
    print('vocab    %d distinct characters' % len(vocab))
    print('length   median %d, p90 %d, max %d'
          % (lens[len(lens) // 2], lens[int(len(lens) * .9)], lens[-1]))
    if ascii_only:
        inside = set(vocab) <= model_vocab
        print('ASCII-folded, inside printable-ASCII vocabulary: %s' % inside)
    print('wrote    %s' % out_path)

    tok_per_step = 8192
    for steps in (16000, 30000):
        print('  %d steps at %d tokens/step = %.1f epochs'
              % (steps, tok_per_step, steps * tok_per_step / len(text)))


def main():
    a = sys.argv[1:]
    if not a or a[0] == 'list':
        print('sources:\n')
        for k, (u, c, f, d) in SOURCES.items():
            print('  %-15s %s' % (k, d))
        print('\n  all             = %s' % ' + '.join(ALL))
        print('\noptions: --out FILE --max-in N --max-out N --truncate '
              '--keep-unicode --no-shuffle')
        return

    names, opt = [], {'out_path': 'corpus.txt', 'max_in': 180,
                      'max_out': 280, 'mode': 'filter', 'ascii_only': True,
                      'shuffle': True}
    i = 0
    while i < len(a):
        k = a[i]
        if k == '--out':
            i += 1
            opt['out_path'] = a[i]
        elif k == '--max-in':
            i += 1
            opt['max_in'] = int(a[i])
        elif k == '--max-out':
            i += 1
            opt['max_out'] = int(a[i])
        elif k == '--truncate':
            opt['mode'] = 'truncate'
        elif k == '--keep-unicode':
            opt['ascii_only'] = False
        elif k == '--no-shuffle':
            opt['shuffle'] = False
        elif k == 'all':
            names.extend(ALL)
        elif k in SOURCES:
            names.append(k)
        else:
            raise SystemExit('unknown source or option %r (try: list)' % k)
        i += 1

    if not names:
        raise SystemExit('no sources given (try: list)')
    build(names, **opt)


if __name__ == '__main__':
    main()