"""Formal->informal Indonesian query transformation (Phase 2).

Three transformation arms live under Phase 2 (see CLAUDE.md):
  1. manual   — native-speaker rewrites (data, produced by hand)
  2. llm      — few-shot LLM paraphrase
  3. lexicon  — deterministic lexicon injection   <-- implemented here

Lexicon injection is the deterministic, controllable arm. It substitutes
formal tokens/phrases with colloquial equivalents from an editable mapping
(resources/lexicons/injection_map.tsv) — the inverse direction of
src/data/normalize.py (slang->formal). Being rule-based it is fully
reproducible given a seed, which is what makes it a clean ablation axis:
`rate` controls how aggressively to inform-alize, and alternative choice is
seeded, so re-running yields byte-identical output.
"""
import os
import random
import re

_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "lexicons", "injection_map.tsv",
)


def load_injection_map(path: str = _MAP_PATH) -> dict[str, list[str]]:
    """Parse the editable TSV into {formal: [slang, ...]}, lowercase keys.

    Blank lines and '#' comments are skipped; '|'-separated alternatives keep
    their order (first = default). Identity-only entries are dropped.
    """
    mapping: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "\t" not in line:
                continue
            key, _, rhs = line.partition("\t")
            key = key.strip().lower()
            alts = [a.strip() for a in rhs.split("|") if a.strip()]
            if key and alts and not (len(alts) == 1 and alts[0].lower() == key):
                mapping[key] = alts
    return mapping


def _compile(mapping: dict[str, list[str]]) -> re.Pattern:
    """Whole-word alternation, longest key first so phrases beat their parts."""
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile("|".join(rf"\b{re.escape(k)}\b" for k in keys), re.IGNORECASE)


def inject(
    text: str,
    mapping: dict[str, list[str]],
    pattern: re.Pattern | None = None,
    *,
    seed: int = 42,
    qid: str = "",
    rate: float = 1.0,
    first_only: bool = False,
) -> str:
    """Replace matched formal words with colloquial forms.

    Choices are seeded by (seed, qid, match position) so output is reproducible
    yet varied across positions. `rate` in [0, 1] is the per-match probability of
    actually swapping (rate<1 leaves a controllable fraction formal, so injection
    intensity is tunable). `first_only` forces the default alternative and
    ignores `rate`. Unmatched text passes through unchanged.
    """
    pattern = pattern if pattern is not None else _compile(mapping)

    def repl(m: re.Match) -> str:
        original = m.group(0)
        alts = mapping[original.lower()]
        if first_only:
            return alts[0]
        rng = random.Random(f"{seed}|{qid}|{m.start()}|{original.lower()}")
        if rng.random() >= rate:
            return original
        return rng.choice(alts)

    return pattern.sub(repl, text)


def inject_queries(
    queries: dict[str, str],
    mapping: dict[str, list[str]] | None = None,
    *,
    seed: int = 42,
    rate: float = 1.0,
    first_only: bool = False,
) -> dict[str, str]:
    """Apply lexicon injection to {qid: formal_text}, returning {qid: informal}."""
    mapping = mapping if mapping is not None else load_injection_map()
    pattern = _compile(mapping)
    return {
        qid: inject(
            text, mapping, pattern,
            seed=seed, qid=qid, rate=rate, first_only=first_only,
        )
        for qid, text in queries.items()
    }
