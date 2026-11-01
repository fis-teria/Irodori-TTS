"""Numerical and path-safety checks for the DACVAE round-trip noise diagnosis (no models)."""

import contextlib
import importlib.util
import io
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from singing_vc import codec_diag as cd

SR = 48000
ROOT = Path(__file__).resolve().parents[1]


def harmonic(seconds=1.0, f0=220.0, level=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 30) if f0 * k < 20000)
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)  # 20 ms ramps: no broadband onset clicks
    return level * fade * tone / np.max(np.abs(tone))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GainTests(unittest.TestCase):
    def test_boost_limited_by_peak_cap(self):
        audio = np.array([0.0, 0.2, -0.4])
        plan = cd.peak_safe_gain_db(audio, 12.0, peak_cap=0.95)
        self.assertTrue(plan["peak_limited"])
        self.assertAlmostEqual(plan["conditioned_peak"], 0.95, places=9)
        self.assertAlmostEqual(plan["gain_db"], 20 * math.log10(0.95 / 0.4), places=9)

    def test_input_above_cap_is_attenuated_not_left_over_cap(self):
        plan = cd.peak_safe_gain_db(np.array([0.99, -0.5]), 6.0, peak_cap=0.95)
        self.assertLess(plan["gain_db"], 0.0)
        self.assertLessEqual(plan["conditioned_peak"], 0.95 + 1e-12)

    def test_silence_gets_requested_gain(self):
        plan = cd.peak_safe_gain_db(np.zeros(10), 6.0)
        self.assertEqual(plan["gain_db"], 6.0)
        self.assertFalse(plan["peak_limited"])

    def test_restore_inverts_conditioning_and_scalar_gain_recovers_it(self):
        x = harmonic(0.2)
        y = (x * 10 ** (9.3 / 20)).astype(np.float32)
        self.assertAlmostEqual(cd.scalar_gain_db(x, y), 9.3, places=4)
        np.testing.assert_allclose(cd.restore_gain(y, 9.3), x, atol=1e-6)


class AlignmentTests(unittest.TestCase):
    def test_lag_and_crop(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(4000)
        late = np.concatenate([np.zeros(37), x])
        self.assertEqual(cd.best_lag(x, late, 100), 37)
        self.assertEqual(cd.best_lag(late, x, 100), -37)
        r, d = cd.align(x, late, 37)
        np.testing.assert_allclose(r, d)


class CompareTests(unittest.TestCase):
    def setUp(self):
        gap = np.zeros(SR // 2)
        self.ref = np.concatenate([gap, harmonic(1.0), gap, harmonic(1.0, f0=330.0), gap])

    def test_identical_signal(self):
        result = cd.compare(self.ref, self.ref.copy(), SR)
        self.assertEqual(result["lag_samples"], 0)
        self.assertAlmostEqual(result["fit_gain_db"], 0.0, places=6)
        self.assertLess(result["waveform_error_ratio_db"], -90)
        self.assertAlmostEqual(result["active"]["log_spectral_distance_db"], 0.0, places=6)
        for band, value in result["active"]["reference_contrast_db"].items():
            self.assertAlmostEqual(result["active"]["decoded_contrast_db"][band], value, places=6)

    def test_pure_gain_is_gain_not_error(self):
        result = cd.compare(self.ref, 0.5 * self.ref, SR)
        self.assertAlmostEqual(result["fit_gain_db"], 20 * math.log10(0.5), places=4)
        self.assertLess(result["waveform_error_ratio_db"], -90)
        # Relative floors are unchanged by a gain; absolute levels move by it.
        self.assertAlmostEqual(result["active"]["level_excess_db"], 20 * math.log10(0.5), places=3)

    def test_noise_between_harmonics_lowers_contrast_and_raises_gap_floor(self):
        rng = np.random.default_rng(1)
        noisy = self.ref + 3e-3 * rng.standard_normal(len(self.ref))
        result = cd.compare(self.ref, noisy, SR)
        active = result["active"]
        for band in active["reference_contrast_db"]:
            self.assertLess(active["decoded_contrast_db"][band], active["reference_contrast_db"][band] - 1.0)
        quiet = result["quiet"]
        self.assertGreater(quiet["decoded_level_rel_active_db"], quiet["reference_level_rel_active_db"] + 20)

    def test_classes_without_frames_are_unavailable(self):
        loud = harmonic(2.0)
        result = cd.compare(loud, loud.copy(), SR)
        self.assertEqual(result["frames"]["quiet"], 0)
        self.assertIsNone(result["quiet"])

    def test_band_above_source_nyquist_sees_generated_energy(self):
        t = np.arange(len(self.ref)) / SR
        extra = self.ref + 1e-3 * np.sin(2 * np.pi * 23000 * t)
        result = cd.compare(self.ref, extra, SR)
        bands = result["active"]
        self.assertGreater(bands["decoded_bands_db"]["22050-24000"], bands["reference_bands_db"]["22050-24000"] + 40)


class PathSafetyTests(unittest.TestCase):
    def test_outputs_never_touch_inputs_or_escape_root(self):
        diag = load_script("codec_noise_diagnosis")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "in.wav"
            source.write_bytes(b"x")
            root = tmp / "run"
            root.mkdir()
            link = root / "link.wav"
            os.link(source, link)  # hardlink inside the run root
            with self.assertRaises(diag.PathError):
                diag.check_output(link, root, [source])
            with self.assertRaises(diag.PathError):
                diag.check_output(tmp / "outside.wav", root, [source])
            with self.assertRaises(diag.PathError):
                diag.check_root(tmp, [source])  # root containing an input
            diag.check_output(root / "new.wav", root, [source])  # fresh path is fine
            self.assertEqual(source.read_bytes(), b"x")

    def test_roundtrip_cli_rejects_output_over_input_before_loading_codec(self):
        rt = load_script("dacvae_roundtrip")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            source.write_bytes(b"x")
            alias = Path(tmp) / "alias.wav"
            os.link(source, alias)
            for argv in (["--input", str(source), "--output", str(alias), "--json", str(Path(tmp) / "r.json")],
                         ["--input", str(source), "--output", str(Path(tmp) / "o.wav"), "--json", str(source)]):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    rt.main(argv)
            self.assertEqual(source.read_bytes(), b"x")


if __name__ == "__main__":
    unittest.main()
