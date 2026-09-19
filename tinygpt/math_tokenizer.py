# coding: utf-8
'''
A fixed, hand-written tokenizer for a generated math corpus.

Unlike the character tokenizer, this one's vocabulary does not depend on
the corpus at all: it is a closed list of lexemes the generator is known to
emit. That has two consequences worth understanding before training on it:

  - The vocabulary is fixed, so every checkpoint trained with it shares one
    id mapping. Editing LEXEMES or SYMBOLS after training invalidates every
    checkpoint trained before the edit, which is why the mapping is stored
    in the checkpoint and verified on load (see tokenizer.from_vocab).

  - strict=True raises on anything outside that closed list, rather than
    quietly dropping it. For a generated corpus that is the desired
    behaviour: unknown text means the generator emitted something the
    tokenizer was never taught, and silently mapping it to <unk> would hide
    the bug until the samples came out wrong.

<pad>, <bos> and <eos> exist in the vocabulary but the flat-stream trainer
in train.py never emits them: it concatenates the whole corpus and slices
random windows, so there are no sequence boundaries to mark. Their
embedding rows stay at their initial values. encode(add_bos=..., add_eos=...)
is there for a padded/batched trainer, which this repository does not have.
'''

from __future__ import annotations


class MathCorpusTokenizer:
    """
    Fixed tokenizer for the math_curriculum_generator corpus.

    Design:
      - lowercase everything
      - ignore spaces/tabs completely
      - preserve newlines as <nl>
      - common generator phrases become single tokens
      - common words become single tokens
      - numbers are digit-level: 123 -> "1", "2", "3"
      - math notation is character-level
      - strict=True raises on any language the tokenizer does not know

    Example:

        U: Solve for x: 3(x - 7) - 8 = -23

    becomes:

        <u><solve_for>x:3(x-7)-8=-23
    """

    # Name under which this tokenizer is stored in a checkpoint.
    name = 'math'

    # Decoded sequences that mean "this turn is over". The model emits a
    # newline before the next speaker marker, exactly as the corpus does.
    stops = ('<nl><u>', '<nl><a>')

    SPECIAL_TOKENS = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<nl>",
    )

    # ------------------------------------------------------------------
    # Generator language
    #
    # Keys are literal lowercase strings emitted by the generator.
    # Values are the corresponding vocabulary tokens.
    #
    # Longer matches are tried first automatically.
    # ------------------------------------------------------------------

    LEXEMES = {
        # Corpus structure
        "u:": "<u>",
        "a:": "<a>",

        # Important multiword concepts
        "solve for": "<solve_for>",
        "final answer": "<final_answer>",
        "both sides": "<both_sides>",
        "order of operations": "<order_of_operations>",
        "least common denominator": "<least_common_denominator>",
        "signed integers": "<signed_integers>",
        "groups of": "<groups_of>",
        "same as": "<same_as>",
        "numerators and denominators": "<numerators_and_denominators>",
        "its reciprocal": "<its_reciprocal>",
        "constant terms": "<constant_terms>",
        "zero-product property": "<zero_product_property>",
        "each equation": "<each_equation>",

        # Single words used by the generator
        "compute": "<compute>",
        "step": "<step>",

        "add": "<add>",
        "and": "<and>",

        "subtract": "<subtract>",
        "subtracting": "<subtracting>",
        "from": "<from>",

        "multiply": "<multiply>",
        "multiplying": "<multiplying>",
        "by": "<by>",

        "divide": "<divide>",

        "contains": "<contains>",
        "so": "<so>",

        "combine": "<combine>",
        "constants": "<constants>",

        "the": "<the>",
        "is": "<is>",

        "adding": "<adding>",

        "use": "<use>",
        "first": "<first>",
        "now": "<now>",

        "of": "<of>",

        "rewrite": "<rewrite>",
        "fractions": "<fractions>",
        "numerators": "<numerators>",

        "simplify": "<simplify>",
        "to": "<to>",

        # Keep article "a" separate from corpus marker <a>
        "a": "<a_word>",
        "fraction": "<fraction>",

        "distribute": "<distribute>",

        "cancel": "<cancel>",
        "leaving": "<leaving>",

        "find": "<find>",
        "two": "<two>",
        "numbers": "<numbers>",
        "whose": "<whose>",
        "product": "<product>",
        "sum": "<sum>",

        "factor": "<factor>",
        "quadratic": "<quadratic>",

        "solve": "<solve>",
        "or": "<or>",
    }

    # Numbers deliberately remain digit-level.
    SYMBOLS = tuple("0123456789") + (
        "x",
        "+",
        "-",
        "×",
        "÷",
        "=",
        "/",
        "(",
        ")",
        ":",
        ".",
        ",",
        "{",
        "}",
        "^",
    )

    def __init__(self, strict: bool = True):
        self.strict = strict

        # Keep vocabulary deterministic.
        lexeme_tokens = list(dict.fromkeys(self.LEXEMES.values()))

        self.vocab = (
            list(self.SPECIAL_TOKENS)
            + lexeme_tokens
            + list(self.SYMBOLS)
        )

        self.token_to_id = {
            token: i
            for i, token in enumerate(self.vocab)
        }

        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }

        # Greedy longest-match tokenization.
        self._surfaces = sorted(
            self.LEXEMES,
            key=len,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Useful properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, and normalize line endings.

        Everything in the language is case-insensitive, and \\r\\n must not
        become two separate <nl> tokens.
        """

        text = text.lower()

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return text

    @staticmethod
    def _boundary_ok(
        text: str,
        start: int,
        surface: str,
    ) -> bool:
        """
        Prevent matching tokens inside larger words.

        For example:
            "add" should not accidentally match inside "adding".
        """

        end = start + len(surface)

        if surface[0].isalnum():
            if start > 0:
                previous = text[start - 1]

                if previous.isalnum() or previous == "_":
                    return False

        if surface[-1].isalnum():
            if end < len(text):
                following = text[end]

                if following.isalnum() or following == "_":
                    return False

        return True

    def _match_lexeme(
        self,
        text: str,
        i: int,
    ) -> str | None:
        """
        Longest generator lexeme starting at offset i, or None.

        Longest strings are attempted first, so:

            solve for

        wins before:

            solve
        """

        for surface in self._surfaces:
            if not text.startswith(surface, i):
                continue

            if not self._boundary_ok(
                text,
                i,
                surface,
            ):
                continue

            return surface

        return None

    def _scan(self, text: str):
        """
        Walk normalized text, yielding (kind, value, offset).

        kind is "token" for anything in the vocabulary and "unknown" for a
        character the tokenizer has no rule for. tokenize() and unknown()
        share this one scan so the two can never disagree about what this
        tokenizer understands; only their policy for "unknown" differs.
        """

        text = self._normalize(text)

        i = 0

        while i < len(text):
            ch = text[i]

            # ----------------------------------------------------------
            # Spaces carry no information in this corpus.
            # ----------------------------------------------------------

            if ch in " \t":
                i += 1
                continue

            # ----------------------------------------------------------
            # Newlines do matter.
            # ----------------------------------------------------------

            if ch == "\n":
                yield ("token", "<nl>", i)
                i += 1
                continue

            # ----------------------------------------------------------
            # Generator language, longest match first.
            # ----------------------------------------------------------

            surface = self._match_lexeme(text, i)

            if surface is not None:
                yield ("token", self.LEXEMES[surface], i)
                i += len(surface)
                continue

            # ----------------------------------------------------------
            # Digits / variables / math symbols
            # ----------------------------------------------------------

            if ch in self.SYMBOLS:
                yield ("token", ch, i)
                i += 1
                continue

            # ----------------------------------------------------------
            # Unknown language
            # ----------------------------------------------------------

            yield ("unknown", ch, i)
            i += 1

    def tokenize(self, text: str) -> list[str]:
        """
        Convert text into vocabulary tokens.

        Raises ValueError on unknown language when strict, otherwise emits
        <unk> in its place.
        """

        tokens = []

        normalized = None

        for kind, value, offset in self._scan(text):
            if kind == "token":
                tokens.append(value)
                continue

            if self.strict:
                if normalized is None:
                    normalized = self._normalize(text)

                start = max(0, offset - 20)
                end = min(len(normalized), offset + 40)

                context = normalized[start:end]

                raise ValueError(
                    f"Unknown text at offset {offset}: "
                    f"{normalized[offset:offset + 20]!r}\n"
                    f"Context: {context!r}"
                )

            tokens.append("<unk>")

        return tokens

    def unknown(self, text: str) -> list[str]:
        """
        Distinct characters this tokenizer has no rule for, first seen
        first. Empty means the text tokenizes cleanly even when strict.
        """

        seen = {}

        for kind, value, _ in self._scan(text):
            if kind == "unknown":
                seen[value] = True

        return list(seen)

    # ------------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:

        tokens = self.tokenize(text)

        if add_bos:
            tokens.insert(0, "<bos>")

        if add_eos:
            tokens.append("<eos>")

        return [
            self.token_to_id[token]
            for token in tokens
        ]

    def ids_to_tokens(
        self,
        ids: list[int],
    ) -> list[str]:

        return [
            self.id_to_token[token_id]
            for token_id in ids
        ]

    def decode(self, ids: list[int]) -> str:
        """
        Ids back to text, in the same no-separator form as compact().

        This is not an inverse of encode(): spacing and letter case were
        discarded on the way in and cannot be recovered. What comes back is
        exactly what the model sees, which is what you want when reading
        samples.
        """

        return "".join(
            self.ids_to_tokens(ids)
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def compact(self, text: str) -> str:
        """
        Human-readable visualization of exactly what the tokenizer sees.

        No separator is inserted between tokens.
        """

        return "".join(
            self.tokenize(text)
        )

    def encode_example(
        self,
        problem: str,
        target: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """
        Encode one problem/answer pair directly.

        Avoids needing to construct the corpus string elsewhere.
        """

        text = (
            f"U: {problem}\n"
            f"A: {target}"
        )

        return self.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
        )
