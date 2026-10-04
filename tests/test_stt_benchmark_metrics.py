from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "stt"))

from metrics import glossary_recall, normalize, realtime_factor, word_error_rate  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_normalize_drops_punctuation_and_case(self) -> None:
        self.assertEqual(normalize("Hello, World! It's Mal-Thorn."), ["hello", "world", "it's", "mal", "thorn"])

    def test_wer_counts_each_error_kind(self) -> None:
        result = word_error_rate("the quick brown fox", "the slow brown fox jumps")
        self.assertEqual((result.substitutions, result.deletions, result.insertions), (1, 0, 1))
        self.assertAlmostEqual(result.wer, 0.5)

    def test_wer_deletion_and_empty_inputs(self) -> None:
        self.assertEqual(word_error_rate("a b c", "a c").deletions, 1)
        self.assertEqual(word_error_rate("a b", "").wer, 1.0)
        self.assertEqual(word_error_rate("", "").wer, 0.0)

    def test_glossary_recall_scores_only_terms_in_reference(self) -> None:
        reference = "Strahd met Ireena. Strahd left."
        hypothesis = "Strad met Ireena. Strahd left."
        scores = glossary_recall(["Strahd", "Ireena", "Barovia"], reference, hypothesis)
        self.assertEqual(scores, {"Strahd": (1, 2), "Ireena": (1, 1)})

    def test_glossary_multiword_term(self) -> None:
        scores = glossary_recall(["Vallaki Keep"], "at Vallaki Keep", "at valaki keep")
        self.assertEqual(scores, {"Vallaki Keep": (0, 1)})

    def test_realtime_factor(self) -> None:
        self.assertEqual(realtime_factor(600, 100), 6.0)
        self.assertEqual(realtime_factor(600, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
