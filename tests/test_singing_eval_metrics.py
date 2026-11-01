"""Numerical checks for the reference-comparison metrics and the seeded Seed-VC wrapper (no models)."""

import math
import unittest

import numpy as np

from singing_vc import eval_metrics as em
from singing_vc import seeded_seedvc

SR = 16000


def tone_with_gap(level=0.1, sr=SR):
    t = np.arange(sr) / sr
    tone = level * np.sin(2 * np.pi * 220 * t)
    return np.concatenate([tone, np.zeros(sr), tone])  # 1 s on, 1 s silent, 1 s on


class ActivityTests(unittest.TestCase):
    def test_frame_levels_and_active_mask(self):
        audio = tone_with_gap()
        levels = em.frame_rms_db(audio, SR)
        self.assertEqual(len(levels), 300)
        # sine RMS = amplitude / sqrt(2)
        self.assertAlmostEqual(float(np.median(levels[10:90])), 20 * math.log10(0.1 / math.sqrt(2)), delta=0.2)
        mask = em.active_mask(levels)
        self.assertGreater(mask[:100].mean(), 0.95)
        self.assertLess(mask[110:190].mean(), 0.01)

    def test_active_rms_ignores_silence(self):
        with_gap = em.active_rms_db(tone_with_gap(), SR)
        without = em.active_rms_db(tone_with_gap()[:SR], SR)
        self.assertAlmostEqual(with_gap, without, delta=0.3)
        self.assertEqual(em.active_rms_db(np.zeros(SR), SR), -math.inf)

    def test_best_window_prefers_active_region(self):
        mask = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=bool)
        self.assertEqual(em.best_window_start(mask, 3), 2)
        self.assertEqual(em.best_window_start(mask, 20), 0)

    def test_select_windows_are_disjoint_and_active(self):
        mask = np.zeros(1000, dtype=bool)
        mask[100:600] = True
        windows = em.select_windows(mask, 200, 0.6)
        self.assertEqual(windows, [(20, 220), (220, 420), (420, 620)])
        for start, end in windows:
            self.assertGreaterEqual(mask[start:end].mean(), 0.6)
        self.assertEqual(em.select_windows(mask[:100], 200), [])


class GainTests(unittest.TestCase):
    def test_matched_gain_respects_peak_ceiling(self):
        free = em.matched_gain(-30.0, 0.1, target_db=-24.0, peak_ceiling_db=-1.0)
        self.assertAlmostEqual(free["gain_db"], 6.0)
        self.assertFalse(free["peak_limited"])
        limited = em.matched_gain(-30.0, 0.8, target_db=-24.0, peak_ceiling_db=-1.0)
        self.assertTrue(limited["peak_limited"])
        self.assertAlmostEqual(limited["achieved_peak_db"], -1.0)

    def test_attenuation_only_never_boosts_or_clips(self):
        levels, peaks = [-14.0, -19.0, -25.0], [0.99, 0.6, 0.2]
        common, gains = em.attenuation_only_gains(levels, peaks, -1.0)
        self.assertTrue(all(g <= 1e-9 for g in gains))
        for level, peak, gain in zip(levels, peaks, gains):
            self.assertAlmostEqual(level + gain, common)
            self.assertLessEqual(em.db(peak) + gain, -1.0 + 1e-9)
        self.assertAlmostEqual(common, -25.0)


class EmbeddingTests(unittest.TestCase):
    def test_empty_or_invalid_embeddings_are_not_reported_as_similarity(self):
        for values in (np.empty((0, 3)), np.array([[np.nan, 0, 1]]), np.zeros((1, 3))):
            with self.assertRaises(ValueError):
                em.centroid(values)

    def test_centroid_weights_windows_equally(self):
        windows = np.array([[10.0, 0.0], [0.0, 0.1]])  # very different norms
        c = em.centroid(windows)
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0)
        self.assertAlmostEqual(c[0], c[1])
        self.assertAlmostEqual(em.cosine(np.array([1.0, 0]), np.array([0, 2.0])), 0.0)


class F0Tests(unittest.TestCase):
    def test_truncated_or_invalid_output_still_uses_entire_source(self):
        src = np.full(20, 220.0)
        r = em.f0_coverage(src, np.full(10, 220.0))
        self.assertEqual(r["source_voiced_frames"], 20)
        self.assertEqual(r["missing_output_tail_frames"], 10)
        self.assertEqual(r["within_50c_of_source_voiced"], 0.5)
        self.assertEqual(r["common_within_50c"], 1.0)
        invalid = em.f0_coverage(src, np.array([0.0, -220.0, np.inf, np.nan] + [220.0] * 16))
        self.assertEqual(invalid["within_50c_of_source_voiced"], 0.8)
        self.assertEqual(em.f0_coverage(src, np.full(30, 220.0))["within_50c_of_source_voiced"], 1.0)

    def test_missing_output_counts_as_failure(self):
        src = np.array([200.0] * 8 + [np.nan] * 2 + [200.0] * 10)
        out = src.copy()
        out[:4] = np.nan                    # 4 source-voiced frames lost
        out[4:6] = 200.0 * 2 ** (100 / 1200)  # 2 frames 100 cents sharp
        r = em.f0_coverage(src, out)
        self.assertEqual(r["source_voiced_frames"], 18)
        self.assertEqual(r["common_voiced_frames"], 14)
        self.assertAlmostEqual(r["output_valid_on_source_voiced"], 14 / 18)
        self.assertAlmostEqual(r["within_50c_of_source_voiced"], 12 / 18)
        self.assertAlmostEqual(r["common_within_50c"], 12 / 14)
        self.assertEqual(r["octave_error_of_source_voiced"], 0.0)

    def test_shift_and_octave(self):
        src = np.full(20, 220.0)
        out = np.full(20, 440.0)
        r = em.f0_coverage(src, out, semitone_shift=0)
        self.assertEqual(r["octave_error_of_source_voiced"], 1.0)
        self.assertEqual(em.f0_coverage(src, out, semitone_shift=12)["within_50c_of_source_voiced"], 1.0)
        self.assertIsNone(r["common_log_f0_correlation"])  # flat contour


class EnvelopeAndTextTests(unittest.TestCase):
    def test_envelope_lag_recovers_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=400)
        b = np.concatenate([np.zeros(3), a[:-3]])  # b is a delayed by 3 frames
        r = em.envelope_lag(a, b, 10)
        self.assertEqual(r["best_lag_frames"], 3)
        self.assertGreater(r["corr_at_best"], 0.99)
        self.assertLess(r["corr_at_zero"], 0.3)

    def test_cer(self):
        self.assertEqual(em.character_error_rate("あいう", "あいう"), 0.0)
        self.assertAlmostEqual(em.character_error_rate("あいうえ", "あいえ"), 0.25)
        self.assertIsNone(em.character_error_rate(" ", "あ"))

    def test_blind_labels_deterministic_permutation(self):
        keys = ["a", "b", "c", "d"]
        first = em.blind_labels(keys, 7)
        self.assertEqual(first, em.blind_labels(keys, 7))
        self.assertNotEqual([first[k] for k in keys], ["X01", "X02", "X03", "X04"])  # actually shuffled
        self.assertEqual(sorted(first.values()), ["X01", "X02", "X03", "X04"])


class SeededWrapperTests(unittest.TestCase):
    def test_existing_listening_pack_preserves_user_ratings(self):
        import importlib.util
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        script = Path(__file__).resolve().parents[1] / "scripts" / "singing_reference_comparison.py"
        spec = importlib.util.spec_from_file_location("comparison_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "listening_pack"
            pack.mkdir()
            ratings = pack / "my_ratings.txt"
            ratings.write_text("X01: 5", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.stage_pack(root, SimpleNamespace())
            self.assertEqual(ratings.read_text(encoding="utf-8"), "X01: 5")

    def test_split_argv(self):
        own, rest = seeded_seedvc.split_argv(["--seed", "1", "--", "--source", "x.wav"])
        self.assertEqual(own, ["--seed", "1"])
        self.assertEqual(rest, ["--source", "x.wav"])
        with self.assertRaises(SystemExit):
            seeded_seedvc.split_argv(["--seed", "1"])

    def test_seed_everything_repeats_torch_draws(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        seeded_seedvc.seed_everything(3, cudnn_deterministic=False)
        first = torch.randn(4)
        info = seeded_seedvc.seed_everything(3, cudnn_deterministic=False)
        self.assertTrue(torch.equal(first, torch.randn(4)))
        self.assertEqual(info["seed"], 3)


if __name__ == "__main__":
    unittest.main()
