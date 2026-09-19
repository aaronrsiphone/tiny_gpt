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

Which is also how to add a problem kind. Write it using only words already
in math_tokenizer.LEXEMES and the run reports nothing unusual; every kind
below was written that way, so no checkpoint is invalidated by adding them.
The "unused" line printed at the end lists lexemes no kind reaches yet,
which is the honest to-do list. Teaching the tokenizer a *new* word is the
expensive move: it renumbers the vocabulary, and every checkpoint trained
before the edit stops loading (by design -- see tokenizer.from_vocab).

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


def dec(cents):
    '''Hundredths as a decimal string.

    The arithmetic stays in integer hundredths from end to end, so no float
    rounding can reach the corpus: 0.1 + 0.2 must read as 0.30, and a model
    trained on 0.30000000000000004 would be learning a bug.
    '''
    sign = '-' if cents < 0 else ''
    whole, frac = divmod(abs(cents), 100)
    return '%s%d.%02d' % (sign, whole, frac)


def term(n):
    '''"x" or "3x": a coefficient on x, with the redundant 1 dropped.'''
    return 'x' if n == 1 else '%dx' % n


def signed_term(n):
    '''"+ x" or "- 3x": how a coefficient on x reads inside an expression.'''
    return '%s %s' % ('+' if n >= 0 else '-', term(abs(n)))


def frac(num, den):
    '''"3/4", or just "3" once the denominator has cancelled away.'''
    return '%d' % num if den == 1 else '%d/%d' % (num, den)


def move_constant(k, coeff, rhs):
    '''The step that clears a constant from the left side.

    Written once because four problem kinds need it, and the wording has to
    follow the sign: you subtract a positive constant and add a negative
    one. Returns the step text.
    '''
    verb = ('Subtract %d from both sides' % k) if k > 0 else \
           ('Add %d to both sides' % -k)
    return '%s, so %s = %d' % (verb, term(coeff), rhs - k)


def reduce_steps(num, den, step):
    '''The cancelling steps for num/den, numbered from `step`.

    Every fraction kind ends through here, for two reasons: the wording of
    "contains a factor of g" and "cancel the g" is then identical wherever
    it appears, and no kind can accidentally stop on an answer that still
    reduces. Returns (steps, final text).
    '''
    g = gcd(num, den)
    if g == 1:
        return [], frac(num, den)
    return ([
        'Step %d: The fraction contains a factor of %d' % (step, g),
        'Step %d: Cancel the %d, leaving %s'
        % (step + 1, g, frac(num // g, den // g)),
    ], frac(num // g, den // g))


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
    steps = [
        'Step 1: Find the least common denominator of %d and %d, so use %d'
        % (a, b, l),
        'Step 2: Rewrite fractions, so %d/%d + %d/%d'
        % (p * (l // a), l, q * (l // b), l),
        'Step 3: Add numerators, so %s' % frac(num, l),
    ]
    tail, final = reduce_steps(num, l, 4)
    return problem, '\n'.join(steps + tail + ['Final answer: %s' % final])


def divide_fractions(rng):
    '''a/b divided by c/d'''
    a, b = rng.randint(1, 9), rng.randint(2, 9)
    c, d = rng.randint(1, 9), rng.randint(2, 9)
    num, den = a * d, b * c

    problem = 'Compute %d/%d ÷ %d/%d' % (a, b, c, d)
    steps = [
        'Step 1: Multiply by its reciprocal, so %d/%d × %s'
        % (a, b, frac(d, c)),
        'Step 2: Multiply numerators and denominators, so %s'
        % frac(num, den),
    ]
    tail, final = reduce_steps(num, den, 3)
    return problem, '\n'.join(steps + tail + ['Final answer: %s' % final])


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

    problem = 'Solve x^2 %s %s = 0' % (signed_term(b), signed(c))
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


def like_terms(rng):
    '''a*x + c*x + b + d = e, collecting terms before solving'''
    x = rng.randint(-9, 9)
    a = rng.randint(2, 9)
    c = rng.randint(2, 9)
    b = rng.randint(-9, 9) or 1
    d = rng.randint(-9, 9) or 2
    while b + d == 0:
        d = rng.randint(-9, 9) or 2
    coeff, k = a + c, b + d
    e = coeff * x + k

    problem = 'Solve for x: %s %s %s %s = %d' \
        % (term(a), signed_term(c), signed(b), signed(d), e)
    answer = '\n'.join([
        'Step 1: Add %dx and %dx, so %dx %s %s = %d'
        % (a, c, coeff, signed(b), signed(d), e),
        'Step 2: Combine constant terms, so %dx %s = %d'
        % (coeff, signed(k), e),
        'Step 3: %s' % move_constant(k, coeff, e),
        'Step 4: Now divide both sides by %d, so x = %d' % (coeff, x),
        'Final answer: x = %d' % x,
    ])
    return problem, answer


def both_sides(rng):
    '''a*x + b = c*x + d, the variable on both sides'''
    x = rng.randint(-9, 9)
    c = rng.randint(1, 7)
    a = c + rng.randint(2, 9)
    coeff = a - c
    b = rng.randint(-9, 9) or 3
    d = coeff * x + b
    while d == 0:
        b = rng.randint(-9, 9) or 3
        d = coeff * x + b

    problem = 'Solve for x: %s %s = %s %s' \
        % (term(a), signed(b), term(c), signed(d))
    answer = '\n'.join([
        'Step 1: Subtract %s from both sides, leaving %s %s = %d'
        % (term(c), term(coeff), signed(b), d),
        'Step 2: %s' % move_constant(b, coeff, d),
        'Step 3: Now divide both sides by %d, so x = %d' % (coeff, x),
        'Final answer: x = %d' % x,
    ])
    return problem, answer


def reduce_fraction(rng):
    '''Cancel the common factor out of g*p / g*q.

    p and q are coprime, so g really is the whole common factor and one
    cancelling step finishes the job.
    '''
    g = rng.randint(2, 9)
    p, q = 1, 1
    while gcd(p, q) != 1 or p == q:
        p, q = rng.randint(1, 9), rng.randint(2, 9)

    problem = 'Simplify %d/%d' % (g * p, g * q)
    steps, final = reduce_steps(g * p, g * q, 1)
    return problem, '\n'.join(steps + ['Final answer: %s' % final])


def groups_of(rng):
    '''Multiplication as repeated addition, kept short enough to read'''
    a = rng.randint(2, 5)
    b = rng.randint(2, 9)

    problem = 'Compute %d × %d' % (a, b)
    answer = '\n'.join([
        'Step 1: %d × %d is the same as %d groups of %d' % (a, b, a, b),
        'Step 2: Adding %s, so %d' % (' and '.join([str(b)] * a), a * b),
        'Final answer: %d' % (a * b),
    ])
    return problem, answer


def subtract_negative(rng):
    '''a - -b, the sign rule stated explicitly'''
    a = rng.randint(1, 20)
    b = rng.randint(1, 20)

    problem = 'Compute %d - -%d' % (a, b)
    answer = '\n'.join([
        'Step 1: Subtracting -%d is the same as adding %d' % (b, b),
        'Step 2: Add %d + %d = %d' % (a, b, a + b),
        'Final answer: %d' % (a + b),
    ])
    return problem, answer


def multiply_fractions(rng):
    '''p/q x r/s, reduced when it needs reducing'''
    p, q = rng.randint(1, 9), rng.randint(2, 9)
    r, s = rng.randint(1, 9), rng.randint(2, 9)
    num, den = p * r, q * s

    problem = 'Compute %d/%d × %d/%d' % (p, q, r, s)
    steps = ['Step 1: Multiplying numerators and denominators, so %s'
             % frac(num, den)]
    tail, final = reduce_steps(num, den, 2)
    return problem, '\n'.join(steps + tail + ['Final answer: %s' % final])


def decimals(rng):
    '''Decimal addition and subtraction, two places, exact'''
    a = rng.randint(100, 5000)
    b = rng.randint(100, 5000)

    if rng.random() < 0.5:
        problem = 'Compute %s + %s' % (dec(a), dec(b))
        step = 'Step 1: Add %s + %s = %s' % (dec(a), dec(b), dec(a + b))
        total = a + b
    else:
        a, b = max(a, b), min(a, b)
        problem = 'Compute %s - %s' % (dec(a), dec(b))
        step = 'Step 1: Subtract %s from %s, so %s' \
            % (dec(b), dec(a), dec(a - b))
        total = a - b

    return problem, '\n'.join([step, 'Final answer: %s' % dec(total)])


def difference_of_squares(rng):
    '''x^2 = r^2, answered as a solution set'''
    r = rng.randint(2, 12)
    sq = r * r

    problem = 'Solve x^2 = %d' % sq
    answer = '\n'.join([
        'Step 1: Subtract %d from both sides, leaving x^2 - %d = 0'
        % (sq, sq),
        'Step 2: Find two numbers whose product is -%d and whose sum is 0, '
        'so use %d and -%d' % (sq, r, r),
        'Step 3: Factor the quadratic, so (x + %d)(x - %d) = 0' % (r, r),
        'Step 4: Use the zero-product property, so solve each equation',
        'Final answer: x = {%d, -%d}' % (r, r),
    ])
    return problem, answer


KINDS = {
    'linear': (two_step, 'a*x + b = c, two steps'),
    'terms': (like_terms, 'collect a*x + c*x and the constant terms first'),
    'sides': (both_sides, 'a*x + b = c*x + d, variable on both sides'),
    'distribute': (distribute, 'a(x - b) - c = d, four steps'),
    'order': (order_of_ops, 'order of operations on a + b x (c - d)'),
    'fractions': (add_fractions, 'p/a + q/b via a common denominator'),
    'multiply': (multiply_fractions, 'p/q x r/s, reduced if needed'),
    'divide': (divide_fractions, 'a/b divided by c/d, via the reciprocal'),
    'reduce': (reduce_fraction, 'cancel the common factor out of a fraction'),
    'groups': (groups_of, 'multiplication as repeated addition'),
    'signed': (signed_integers, 'signed integer addition, one step'),
    'negatives': (subtract_negative, 'a - -b, subtracting a negative'),
    'decimals': (decimals, 'decimal addition and subtraction, two places'),
    'quadratic': (quadratic, 'factor x^2 + bx + c = 0'),
    'squares': (difference_of_squares, 'x^2 = r^2, answered as a set'),
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

    print('kinds    %s' % ', '.join(kinds))
    print('wrote    %s' % out_path)
    print('%d exchanges, %d chars, %d tokens (%.2f chars/token)'
          % (len(chunks), len(text), len(ids), len(text) / len(ids)))

    # Which tokens never occur is the useful direction to report: an unused
    # lexeme is either a problem kind nobody wrote yet or a word the
    # tokenizer should not be carrying. <pad>/<bos>/<eos>/<unk> are expected
    # here -- the flat-stream trainer has no use for them.
    used = set(ids)
    idle = [t for t in tok.vocab if tok.token_to_id[t] not in used]
    print('vocab    %d of %d tokens occur' % (len(used), tok.vocab_size))
    if idle:
        print('unused   %s' % ' '.join(idle))
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
