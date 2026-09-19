# coding: utf-8
'''
Generate a math worked-solution corpus for the token-level tokenizer.

    python math_corpus.py list                    the problem kinds
    python math_corpus.py --n 4000 --out math.txt
    python math_corpus.py linear quadratic --n 2000

The corpus and tinygpt/math_tokenizer.py are two halves of one contract:
the tokenizer knows a closed list of lexemes, and every line written here
must be inside it. That is checked, not assumed -- the whole corpus is
tokenized with strict=True before it is written, so a template using a word
the tokenizer was never taught fails here rather than at training time.

Unlike direct_corpus.py, nothing is downloaded: the problems are generated,
so the answers are correct by construction and the language stays closed.

Format matches direct_corpus.py, one blank line between exchanges:

    U: Solve for x: 3(x - 7) - 8 = -23
    A: Step 1: Distribute 3, so 3x - 21 - 8 = -23
    ...
    Final answer: x = 2
'''

import random
import sys

from tinygpt.math_tokenizer import MathCorpusTokenizer


def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    return a * b // gcd(a, b)


def signed(n):
    '''" + 5" or " - 5": how a term reads inside an expression.'''
    return '+ %d' % n if n >= 0 else '- %d' % (-n)


# --------------------------------------------------------------- generators
#
# Each returns (problem, answer). Answers are built from the same arithmetic
# that produced the problem, so they are right by construction -- there is no
# solver here to disagree with.


def two_step(rng):
    '''a*x + b = c'''
    x = rng.randint(-9, 9)
    a = rng.randint(2, 9)
    b = rng.randint(-9, 9) or 1
    c = a * x + b

    problem = 'Solve for x: %dx %s = %d' % (a, signed(b), c)
    move = ('Subtract %d from both sides' % b) if b > 0 else \
           ('Add %d to both sides' % -b)
    answer = '\n'.join([
        'Step 1: %s, so %dx = %d' % (move, a, c - b),
        'Step 2: Divide both sides by %d, so x = %d' % (a, x),
        'Final answer: x = %d' % x,
    ])
    return problem, answer


def distribute(rng):
    '''a(x - b) - c = d'''
    x = rng.randint(-9, 9)
    a = rng.randint(2, 9)
    b = rng.randint(1, 9)
    c = rng.randint(1, 9)
    d = a * (x - b) - c

    problem = 'Solve for x: %d(x - %d) - %d = %d' % (a, b, c, d)
    answer = '\n'.join([
        'Step 1: Distribute %d, so %dx - %d - %d = %d'
        % (a, a, a * b, c, d),
        'Step 2: Combine constants, so %dx - %d = %d'
        % (a, a * b + c, d),
        'Step 3: Add %d to both sides, so %dx = %d'
        % (a * b + c, a, d + a * b + c),
        'Step 4: Divide both sides by %d, so x = %d' % (a, x),
        'Final answer: x = %d' % x,
    ])
    return problem, answer


def order_of_ops(rng):
    '''a + b * (c - d)'''
    a = rng.randint(1, 20)
    b = rng.randint(2, 9)
    c = rng.randint(2, 20)
    d = rng.randint(1, c - 1)
    inner = c - d
    prod = b * inner

    problem = 'Compute %d + %d × (%d - %d)' % (a, b, c, d)
    answer = '\n'.join([
        'Step 1: Use order of operations, so first compute %d - %d = %d'
        % (c, d, inner),
        'Step 2: Multiply %d × %d = %d' % (b, inner, prod),
        'Step 3: Add %d + %d = %d' % (a, prod, a + prod),
        'Final answer: %d' % (a + prod),
    ])
    return problem, answer


def add_fractions(rng):
    '''p/a + q/b'''
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    while b == a:
        b = rng.randint(2, 9)
    p = rng.randint(1, a - 1)
    q = rng.randint(1, b - 1)
    l = lcm(a, b)
    num = p * (l // a) + q * (l // b)

    problem = 'Compute %d/%d + %d/%d' % (p, a, q, b)
    answer = '\n'.join([
        'Step 1: Find the least common denominator of %d and %d, so use %d'
        % (a, b, l),
        'Step 2: Rewrite fractions, so %d/%d + %d/%d'
        % (p * (l // a), l, q * (l // b), l),
        'Step 3: Add numerators, so %d/%d' % (num, l),
        'Final answer: %d/%d' % (num, l),
    ])
    return problem, answer


def divide_fractions(rng):
    '''a/b divided by c/d'''
    a, b = rng.randint(1, 9), rng.randint(2, 9)
    c, d = rng.randint(1, 9), rng.randint(2, 9)
    num, den = a * d, b * c
    g = gcd(num, den)

    problem = 'Compute %d/%d ÷ %d/%d' % (a, b, c, d)
    steps = [
        'Step 1: Multiply by its reciprocal, so %d/%d × %d/%d'
        % (a, b, d, c),
        'Step 2: Multiply numerators and denominators, so %d/%d'
        % (num, den),
    ]
    if g > 1:
        steps.append('Step 3: Simplify the fraction, so %d/%d'
                     % (num // g, den // g))
    steps.append('Final answer: %d/%d' % (num // g, den // g))
    return problem, '\n'.join(steps)


def signed_integers(rng):
    '''-a + b, the one-step case'''
    x = -rng.randint(1, 20)
    y = rng.randint(-20, 20)

    problem = 'Compute %d %s' % (x, signed(y))
    answer = '\n'.join([
        'Step 1: Add signed integers, so %d %s = %d' % (x, signed(y), x + y),
        'Final answer: %d' % (x + y),
    ])
    return problem, answer


def quadratic(rng):
    '''x^2 + (r+s)x + r*s = 0, always factorable'''
    r = rng.randint(-9, 9) or 1
    s = rng.randint(-9, 9) or 2
    b, c = r + s, r * s

    problem = 'Solve x^2 %s %s = 0' % (signed(b) + 'x', signed(c))
    answer = '\n'.join([
        'Step 1: Find two numbers whose product is %d and whose sum is %d, '
        'so use %d and %d' % (c, b, r, s),
        'Step 2: Factor the quadratic, so (x %s)(x %s) = 0'
        % (signed(r), signed(s)),
        'Step 3: Use the zero-product property, so solve each equation',
        'Step 4: x = %d or x = %d' % (-r, -s),
        'Final answer: x = %d or x = %d' % (-r, -s),
    ])
    return problem, answer


KINDS = {
    'linear': (two_step, 'a*x + b = c, two steps'),
    'distribute': (distribute, 'a(x - b) - c = d, four steps'),
    'order': (order_of_ops, 'order of operations on a + b x (c - d)'),
    'fractions': (add_fractions, 'p/a + q/b via a common denominator'),
    'divide': (divide_fractions, 'a/b divided by c/d, via the reciprocal'),
    'signed': (signed_integers, 'signed integer addition, one step'),
    'quadratic': (quadratic, 'factor x^2 + bx + c = 0'),
}

ALL = sorted(KINDS)


def build(kinds, n, out_path, seed=0):
    rng = random.Random(seed)
    tok = MathCorpusTokenizer(strict=True)

    chunks = []
    for i in range(n):
        name = kinds[i % len(kinds)]
        problem, answer = KINDS[name][0](rng)
        chunks.append('U: %s\nA: %s' % (problem, answer))

    rng.shuffle(chunks)
    text = '\n\n'.join(chunks) + '\n'

    # The contract check. A template that used a word the tokenizer has
    # never been taught dies here, naming the offending text, instead of
    # silently becoming <unk> or blowing up mid-training.
    miss = tok.unknown(text)
    if miss:
        raise SystemExit(
            'the tokenizer has no rule for %r -- either the template is '
            'wrong or math_tokenizer.LEXEMES needs the word' % ''.join(miss))
    ids = tok.encode(text)

    with open(out_path, 'w') as f:
        f.write(text)

    used = len(set(ids))
    print('kinds    %s' % ', '.join(kinds))
    print('wrote    %s' % out_path)
    print('%d exchanges, %d chars, %d tokens (%.2f chars/token)'
          % (len(chunks), len(text), len(ids), len(text) / len(ids)))
    print('vocab    %d of %d tokens occur' % (used, tok.vocab_size))
    print('\nfirst exchange, as the model sees it:')
    print(tok.compact(chunks[0]))

    tok_per_step = 32 * 256
    print('\n%d steps at %d tokens/step = %.1f epochs'
          % (3600, tok_per_step, 3600 * tok_per_step / len(ids)))


def main():
    a = sys.argv[1:]
    if a and a[0] == 'list':
        print('problem kinds:\n')
        for k in ALL:
            print('  %-11s %s' % (k, KINDS[k][1]))
        print('\n  default    = every kind, round robin')
        print('\noptions: --out FILE --n COUNT --seed S')
        return

    kinds, opt = [], {'out_path': 'math.txt', 'n': 4000, 'seed': 0}
    i = 0
    while i < len(a):
        k = a[i]
        if k == '--out':
            i += 1
            opt['out_path'] = a[i]
        elif k == '--n':
            i += 1
            opt['n'] = int(a[i])
        elif k == '--seed':
            i += 1
            opt['seed'] = int(a[i])
        elif k in KINDS:
            kinds.append(k)
        else:
            raise SystemExit('unknown kind or option %r (try: list)' % k)
        i += 1

    build(kinds or ALL, **opt)


if __name__ == '__main__':
    main()
