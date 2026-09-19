# coding: utf-8
'''
Text <-> integer ids.

The model knows nothing about text. GPT.forward() takes integer ids and
returns a distribution over cfg.vocab of them; whether id 42 means the
letter "q" or the phrase "<least_common_denominator>" is entirely this
module's business. That separation is why a new tokenizer needs no changes
in model.py.

A tokenizer here is any object with:

    name           str, stored in the checkpoint so the right one is rebuilt
    token_to_id    dict[str, int]
    id_to_token    dict[int, str]
    vocab_size     int
    strict         bool: does encode() raise on text it cannot represent?
    stops          tuple[str, ...]: decoded sequences that end a turn
    encode(text)   -> list[int]
    decode(ids)    -> str
    unknown(text)  -> list[str], the pieces it cannot represent

decode(encode(text)) is not required to return text: CharTokenizer round
trips exactly, MathCorpusTokenizer discards spacing and case. What decode()
guarantees is showing what the model actually sees.

Two are provided: CharTokenizer (vocabulary derived from the corpus, one id
per character) and MathCorpusTokenizer (fixed vocabulary, one id per lexeme
-- see math_tokenizer.py).
'''

from tinygpt.data import make_vocab
from tinygpt.math_tokenizer import MathCorpusTokenizer


class CharTokenizer(object):
    '''One id per character, vocabulary taken from the training corpus.

    The original tiny_gpt behaviour, kept as a tokenizer so that the rest of
    the codebase has exactly one way to turn text into ids.

    Not strict: characters outside the vocabulary are dropped rather than
    raising. A character model's vocabulary is whatever the corpus happened
    to contain, so an unseen character in a prompt is ordinary, not a bug --
    the opposite of the fixed-vocabulary case in math_tokenizer.py.
    '''

    name = 'char'
    strict = False

    # A turn ends at a newline followed by the next speaker marker.
    stops = ('\nU:', '\nA:')

    def __init__(self, token_to_id):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: c for c, i in self.token_to_id.items()}

    @classmethod
    def from_text(cls, text):
        stoi, _ = make_vocab(text)
        return cls(stoi)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def encode(self, text):
        stoi = self.token_to_id
        return [stoi[c] for c in text if c in stoi]

    def decode(self, ids):
        itos = self.id_to_token
        return ''.join(itos[i] for i in ids)

    def unknown(self, text):
        return sorted(set(c for c in text if c not in self.token_to_id))


TOKENIZERS = {
    'char': CharTokenizer,
    'math': MathCorpusTokenizer,
}


def build(name, text=None):
    '''Construct a tokenizer by name, for training.

    'char' derives its vocabulary from the corpus and so needs the text.
    Fixed-vocabulary tokenizers ignore it.
    '''
    if name not in TOKENIZERS:
        raise SystemExit('unknown tokenizer %r (choices: %s)'
                         % (name, ', '.join(sorted(TOKENIZERS))))
    if name == 'char':
        if text is None:
            raise SystemExit('the char tokenizer needs the corpus text')
        return CharTokenizer.from_text(text)
    return TOKENIZERS[name]()


def from_vocab(name, token_to_id):
    '''Rebuild the tokenizer a checkpoint was trained with.

    For a corpus-derived vocabulary the stored mapping *is* the tokenizer.
    For a fixed one the mapping is redundant -- and therefore worth
    checking: if LEXEMES or SYMBOLS were edited after this checkpoint was
    trained, its embedding rows no longer line up with the ids this
    tokenizer now produces, and every sample would be quietly wrong.
    '''
    if name == 'char':
        return CharTokenizer(token_to_id)

    tok = build(name)
    if tok.token_to_id != token_to_id:
        raise SystemExit(
            'the %r tokenizer no longer matches this checkpoint: it was '
            'trained with %d tokens and now produces %d. Its vocabulary was '
            'edited after training, so the checkpoint\'s embedding rows no '
            'longer mean what they meant. Retrain, or restore the previous '
            'vocabulary.' % (name, len(token_to_id), tok.vocab_size))
    return tok
