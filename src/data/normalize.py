"""Informal->formal Indonesian normalization via Kamus Alay lexicon lookup.

Phase 1 diagnostic arm: retrieval on normalized informal queries measures how
much of the register gap is pure vocabulary (recoverable by substitution) vs
distribution shift (the residual only representation alignment can address).

Two merged lexicons under resources/lexicons/:
  - new_kamusalay.csv        (Ibrohim & Budi 2019, headerless slang,formal)
  - colloquial-indonesian-lexicon.csv  (Salsabila et al. 2018, with header)
The Salsabila lexicon is loaded second and wins on conflicting slang keys.
"""
import csv
import os
import re

_LEXICON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "lexicons",
)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CHAR_RUN_RE = re.compile(r"(\w)\1{2,}", re.UNICODE)


def load_lexicon(lexicon_dir: str = _LEXICON_DIR) -> dict[str, str]:
    """Merged {slang: formal} map, lowercase keys, identity entries dropped."""
    lex: dict[str, str] = {}

    path = os.path.join(lexicon_dir, "new_kamusalay.csv")
    # File has a few mojibake bytes; decode permissively and skip broken rows.
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and "�" not in row[0] + row[1]:
                _add(lex, row[0], row[1])

    path = os.path.join(lexicon_dir, "colloquial-indonesian-lexicon.csv")
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            _add(lex, row["slang"], row["formal"])

    return lex


def _add(lex: dict[str, str], slang: str, formal: str) -> None:
    slang, formal = slang.strip().lower(), formal.strip().lower()
    if slang and formal and slang != formal:
        lex[slang] = formal


def normalize(text: str, lexicon: dict[str, str]) -> str:
    """Token-level substitution; unknown tokens pass through unchanged.

    Elongations are squeezed before lookup ("bangetttt" -> "banget"): runs of
    3+ identical characters are tried at length 2, then 1. Output is
    space-joined — spacing around punctuation is not restored, which is
    irrelevant to the dense encoder.
    """
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        low = tok.lower()
        for cand in (low, _CHAR_RUN_RE.sub(r"\1\1", low), _CHAR_RUN_RE.sub(r"\1", low)):
            if cand in lexicon:
                out.append(lexicon[cand])
                break
        else:
            out.append(tok)
    return " ".join(out)


def normalize_queries(queries: dict[str, str], lexicon: dict[str, str] | None = None) -> dict[str, str]:
    lexicon = lexicon if lexicon is not None else load_lexicon()
    return {qid: normalize(text, lexicon) for qid, text in queries.items()}
