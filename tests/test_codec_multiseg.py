"""Selection, quiet-span, crop and A/B-assignment rules for the multi-segment codec check (no models)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from singing_vc import codec_diag as cd
from singing_vc import multiseg as ms

SR = 48000
ROOT = Path(__file__).resolve().parents[1]


def tone(seconds, level=0.1, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return level * np.sin(2 * np.pi * 220 * t)


class SelectionTests(unittest.TestCase):
    def test_interval_samples_are_exact_and_bounded(self):
        self.assertEqual(ms.interval_samples(55, 75, 44100, 44100 * 200), (55 * 44100, 75 * 44100))
        with self.assertRaises(ValueError):
            ms.interval_samples(190, 210, 44100, 44100 * 200)
        with self.assertRaises(ValueError):
            ms.interval_samples(10, 10, 44100, 44100 * 200)

    def test_overlap_is_half_open(self):
        self.assertFalse(ms.overlaps((0, 10), (10, 20)))
        self.assertTrue(ms.overlaps((0, 11), (10, 20)))

    def test_activity_flags_silence_and_sparse_audio(self):
        self.assertTrue(ms.activity(np.zeros(SR * 2), SR)["essentially_silent"])
        sparse = np.zeros(SR * 10)
        sparse[: SR] = tone(1, 0.3)  # 10% active
        self.assertTrue(ms.activity(sparse, SR)["essentially_silent"])
        busy = ms.activity(np.stack([tone(4), tone(4)], axis=1), SR)
        self.assertFalse(busy["essentially_silent"])
        self.assertIsNotNone(busy["active_rms_db"])

    def test_alternative_start_skips_taken_and_inactive_and_wraps(self):
        active = {0.0, 40.0, 60.0}
        start = ms.alternative_start(55.0, 20.0, 100.0, [(55.0, 75.0)], lambda s: s in active)
        self.assertEqual(start, 0.0)  # 60 and 40 overlap 55-75; wrap to 0
        self.assertEqual(ms.alternative_start(20.0, 20.0, 100.0, [], lambda s: s in active), 40.0)
        self.assertIsNone(ms.alternative_start(0.0, 20.0, 30.0, [], lambda s: False))


class QuietSpanTests(unittest.TestCase):
    def test_prefers_exact_zero_run_and_erodes(self):
        x = np.concatenate([tone(1), np.zeros(SR), tone(1)])
        span = ms.longest_quiet_span(x, SR)
        self.assertEqual(span["kind"], "exact_zero")
        self.assertEqual((span["start"], span["end"]), (SR + SR // 20, 2 * SR + 1 - SR // 20))  # the next sine starts at exactly 0

    def test_falls_back_to_low_level_frames_and_reports_unavailable(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([tone(1), 1e-5 * rng.standard_normal(SR), tone(1)])  # about -100 dBFS
        span = ms.longest_quiet_span(x, SR)
        self.assertTrue(span["kind"].startswith("frames_below"))
        self.assertTrue(SR <= span["start"] < span["end"] <= 2 * SR + SR // 50)
        self.assertIsNone(ms.longest_quiet_span(tone(2), SR))

    def test_dc_offset_is_separated_from_ac(self):
        x = 1e-3 + 1e-6 * np.sin(np.arange(SR) / 7)
        stats = cd.dc_ac_stats(x)
        self.assertGreater(stats["dc_energy_fraction"], 0.99)
        self.assertLess(stats["centered_ac_rms_dbfs"], stats["total_rms_dbfs"] - 40)


class CropAndPackTests(unittest.TestCase):
    def test_crop_never_pads(self):
        out, info = ms.crop_to_reference(np.ones(110), 100)
        self.assertEqual(len(out), 100)
        self.assertEqual(info["cropped_tail_samples"], 10)
        with self.assertRaises(ValueError):
            ms.crop_to_reference(np.ones(90), 100)

    def test_shared_mask_levels_use_the_source_frames_only(self):
        src = np.concatenate([tone(1, 0.3), np.zeros(SR)])
        noisy = src + np.concatenate([np.zeros(SR), np.full(SR, 0.03)])  # energy only where the source is silent
        quieter = 0.5 * src
        levels = ms.shared_mask_levels(src, [src, noisy, quieter], SR)["levels_db"]
        self.assertAlmostEqual(levels[0], levels[1], delta=0.05)  # silent-part energy is outside the mask
        self.assertAlmostEqual(levels[0] - levels[2], 20 * np.log10(2), delta=1e-6)
        with self.assertRaises(ValueError):
            ms.shared_mask_levels(src, [src[:-10]], SR)
        with self.assertRaises(ValueError):
            ms.shared_mask_levels(np.zeros(SR), [np.zeros(SR)], SR)

    def test_ab_assignment_is_seeded_and_complete(self):
        segs = [f"s{i}" for i in range(20)]
        a = ms.assign_ab(segs, 7)
        self.assertEqual(a, ms.assign_ab(segs, 7))
        for row in a.values():
            self.assertEqual(sorted(row.values()), ["baseline", "candidate"])
        self.assertGreater(len({row["A"] for row in a.values()}), 1)  # not one fixed order


class ScriptPathTests(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location("cmsv", ROOT / "scripts" / "codec_multi_segment_validation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_select_writes_plan_and_bit_exact_excerpts_without_touching_source(self):
        mod = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "song.wav"
            rate = 8000
            audio = np.stack([tone(30, 0.2, rate), tone(30, 0.1, rate)], axis=1)
            audio[5 * rate: 15 * rate] = 0  # 5-15 s digital silence
            sf.write(str(src), audio, rate, subtype="PCM_16")
            before = mod.cnd.sha256(src)
            mod.PREVIOUS_INTERVAL = (25.0, 26.0)
            root = mod.stage_select(mod.parse_args(["select", "--source", str(src), "--root", str(tmp / "run"),
                                                    "--intervals", "1-3", "5-10"]))
            plan = mod.load_plan(root)
            self.assertEqual(mod.cnd.sha256(src), before)
            first, second = plan["segments"]
            self.assertEqual((first["first_sample"], first["end_sample_exclusive"]), (rate, 3 * rate))
            self.assertTrue(second["replaced"])
            self.assertTrue(second["facts_requested"]["essentially_silent"])
            self.assertFalse(second["facts"]["essentially_silent"])
            self.assertEqual(second["seconds"], [15.0, 20.0])  # 5 s grid: 5 and 10 are silent
            self.assertEqual(second["first_sample"], 15 * rate)
            excerpt, _ = sf.read(str(root / first["excerpt"]), dtype="int16")
            original, _ = sf.read(str(src), dtype="int16")
            np.testing.assert_array_equal(excerpt, original[rate: 3 * rate])
            with self.assertRaises(FileExistsError):  # fresh root only
                mod.stage_select(mod.parse_args(["select", "--source", str(src), "--root", str(root)]))

    def test_select_rejects_overlap_with_rated_interval(self):
        mod = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "song.wav"
            sf.write(str(src), tone(40, 0.2, 8000), 8000, subtype="PCM_16")
            with self.assertRaises(SystemExit):
                mod.stage_select(mod.parse_args(["select", "--source", str(src), "--root", str(tmp / "r"),
                                                 "--intervals", "30-38"]))


if __name__ == "__main__":
    unittest.main()
