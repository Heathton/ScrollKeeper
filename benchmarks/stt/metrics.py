"""Pure scoring helpers for the STT benchmark (no third-party dependencies)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_NON_WORD = re.compile(r"[^\w' ]+")


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation (keep apostrophes) and split into words."""
    return _NON_WORD.sub(" ", text.lower().replace("-", " ")).split()


@dataclass
class WerResult:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    ref, hyp = normalize(reference), normalize(hypothesis)
    # dp[j] = (total, subs, dels, ins) for the current reference prefix vs hyp[:j]
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            )
            cur.append(best)
        prev = cur
    _, subs, dels, ins = prev[-1]
    return WerResult(subs, dels, ins, len(ref))


def glossary_recall(glossary: list[str], reference: str, hypothesis: str) -> dict[str, tuple[int, int]]:
    """Per glossary term: (occurrences in the hypothesis, occurrences in the reference).

    Terms may be multi-word. Only terms present in the reference are scored; the
    hit count is capped at the reference count so repeats cannot inflate recall.
    """
    ref_text = " ".join(normalize(reference))
    hyp_text = " ".join(normalize(hypothesis))
    scores: dict[str, tuple[int, int]] = {}
    for term in glossary:
        needle = " ".join(normalize(term))
        if not needle:
            continue
        pattern = re.compile(rf"(?<!\S){re.escape(needle)}(?!\S)")
        expected = len(pattern.findall(ref_text))
        if expected:
            scores[term] = (min(len(pattern.findall(hyp_text)), expected), expected)
    return scores


def realtime_factor(audio_seconds: float, wall_seconds: float) -> float:
    """How many seconds of audio are transcribed per second of wall time."""
    return audio_seconds / wall_seconds if wall_seconds > 0 else 0.0
