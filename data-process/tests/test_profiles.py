"""Profile parameters: defaults, run-config overrides and boundary behaviour."""

from __future__ import annotations

import unittest

from zgcm_data_pipeline.profiles import resolve_profile
from zgcm_data_pipeline.stages.midtrain.datasets.web_knowledge import length_bucket, quality_band
from zgcm_data_pipeline.stages.midtrain.datasets.agent_coding import rough_length_bucket
from zgcm_data_pipeline.stages.posttrain.datasets.dolci_think import has_repetition


class ProfileDefaultsTests(unittest.TestCase):
    """The JSON defaults must equal the constants that used to live in the code."""

    def test_pretrain_defaults(self) -> None:
        self.assertEqual(resolve_profile({}, "pretrain", "finepdfs"), {"lid_threshold": 0.85, "minhash_max": 2.0, "primary_threshold": 0.8})
        self.assertEqual(resolve_profile({}, "pretrain", "olmocr"), {"min_compression_ratio": 0.08, "min_edu_score": 0.6})
        self.assertEqual(
            resolve_profile({}, "pretrain", "gharchive"),
            {"max_file_bytes": 0, "minified_min_chars": 2000, "minified_longest_line": 200000, "minified_few_lines": 10, "minified_average_line": 1000},
        )

    def test_midtrain_defaults(self) -> None:
        self.assertEqual(resolve_profile({}, "midtrain", "nemotron_math"), {"drop_keys": [], "hard_error_threshold": 4.0})
        web = resolve_profile({}, "midtrain", "web_knowledge")
        self.assertEqual(web["default_band"], "lt3")
        self.assertEqual(web["min_quality_band"], "lt3")
        self.assertEqual(web["length_boundaries"], [[16384, "le_16k"], [65536, "gt_16k_le_64k"], [262144, "gt_64k_le_256k"]])
        self.assertEqual(web["overflow_length_label"], "gt_256k")

    def test_posttrain_defaults(self) -> None:
        self.assertEqual(
            resolve_profile({}, "posttrain", "dolci_think"),
            {"strict": True, "repetition_min_words": 120, "repetition_ngram": 5, "repetition_max_ratio": 0.2, "review_rough_tokens": 16000, "max_rough_tokens": 32768},
        )


class OverrideTests(unittest.TestCase):
    def test_run_config_overrides_default_by_profile_prefix(self) -> None:
        resolved = resolve_profile({"finepdfs_lid_threshold": 0.5}, "pretrain", "finepdfs")
        self.assertEqual(resolved["lid_threshold"], 0.5)
        self.assertEqual(resolved["minhash_max"], 2.0)

    def test_override_for_another_profile_is_ignored(self) -> None:
        resolved = resolve_profile({"olmocr_min_edu_score": 0.9}, "pretrain", "finepdfs")
        self.assertNotIn("min_edu_score", resolved)
        self.assertEqual(resolved["primary_threshold"], 0.8)

    def test_resolved_values_are_copies(self) -> None:
        first = resolve_profile({}, "midtrain", "web_knowledge")
        first["length_boundaries"].append([1, "mutated"])
        self.assertEqual(len(resolve_profile({}, "midtrain", "web_knowledge")["length_boundaries"]), 3)


class BoundaryTests(unittest.TestCase):
    def test_quality_bands(self) -> None:
        self.assertEqual(quality_band(4.0), "ge4")
        self.assertEqual(quality_band(3.99), "ge3")
        self.assertEqual(quality_band(3.0), "ge3")
        self.assertEqual(quality_band(2.99), "lt3")
        self.assertEqual(quality_band(-1.0), "lt3")

    def test_length_buckets(self) -> None:
        self.assertIsNone(length_bucket(0))
        self.assertEqual(length_bucket(1), "le_16k")
        self.assertEqual(length_bucket(16384), "le_16k")
        self.assertEqual(length_bucket(16385), "gt_16k_le_64k")
        self.assertEqual(length_bucket(262144), "gt_64k_le_256k")
        self.assertEqual(length_bucket(262145), "gt_256k")

    def test_agent_coding_length_buckets(self) -> None:
        # tokens = int(chars / 3.6): 58985 chars is 16384 tokens, 58986 is 16385.
        self.assertEqual(rough_length_bucket(58985), "le16k")
        self.assertEqual(rough_length_bucket(58986), "gt16k_le64k")
        self.assertEqual(rough_length_bucket(10**9), "gt256k")

    def test_repetition_threshold_and_ngram_size(self) -> None:
        self.assertFalse(has_repetition("a b c " * 10))  # fewer than 120 words
        words = " ".join(f"w{i}" for i in range(200))
        self.assertFalse(has_repetition(words))
        repeated = " ".join(["same"] * 200)
        self.assertTrue(has_repetition(repeated))


if __name__ == "__main__":
    unittest.main()
