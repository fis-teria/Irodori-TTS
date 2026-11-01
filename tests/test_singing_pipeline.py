"""Argument, command and failure-path checks for the singing prototype (no models loaded)."""

import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from singing_vc import audio_tool, cli, pipeline, seedvc_runner
from singing_vc.pipeline import PipelineError, SingingOptions


def _config(f0, sr=44100):
    return {
        "preprocess_params": {"sr": sr},
        "model_params": {
            "length_regulator": {"f0_condition": f0},
            "DiT": {"f0_condition": f0},
            "vocoder": {"type": "bigvgan", "name": "nvidia/bigvgan_v2_44khz_128band_512x"},
        },
    }


def _quiet(_line):
    pass


# Stand-in for upstream inference.py: writes one WAV into --output, like the real script.
FAKE_INFERENCE = """
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "vc_fake.wav").write_bytes(b"converted")
"""
FAKE_TORCH = types.SimpleNamespace(__version__="fake", cuda=types.SimpleNamespace(is_available=lambda: False))


class SingingPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        folder = self.tmp / "dir with spaces と日本語"
        folder.mkdir()
        self.wavs = {name: folder / f"{name} file.v1.wav" for name in ("source", "reference", "speaker")}
        for path in self.wavs.values():
            path.write_bytes(b"RIFF")

    def options(self, **values):
        values.setdefault("source_wav", str(self.wavs["source"]))
        return SingingOptions(**values)

    def test_validate_requires_exactly_one_reference(self):
        both = self.options(reference_wav=str(self.wavs["reference"]), tts_text="こんにちは")
        for options in (self.options(), both):
            self.assertTrue(any("exactly one" in problem for problem in pipeline.validate(options)))
        self.assertEqual(pipeline.validate(self.options(reference_wav=str(self.wavs["reference"]))), [])

    def test_validate_reports_missing_files(self):
        options = self.options(
            source_wav=str(self.tmp / "missing.wav"), tts_text="a",
            tts_lora_adapter=str(self.tmp / "no_adapter"), seedvc_checkpoint=str(self.tmp / "x.pth"),
        )
        problems = "\n".join(pipeline.validate(options))
        for expected in ("source vocal not found", "LoRA adapter directory not found",
                         "Seed-VC checkpoint not found"):
            self.assertIn(expected, problems)

    def test_f0_false_never_falls_back_to_speech_model(self):
        options = self.options(reference_wav=str(self.wavs["reference"]), f0_condition=False)
        self.assertTrue(any("never swapped" in problem for problem in pipeline.validate(options)))

    def test_irodori_command_keeps_paths_with_spaces_as_single_arguments(self):
        options = self.options(tts_text="歌の練習です。", tts_caption="女性の声",
                               tts_speaker_wav=str(self.wavs["speaker"]), tts_seed=7)
        out = self.tmp / "out dir" / "reference.wav"
        command = pipeline.irodori_command(options, out)

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(command[:2], [str(pipeline.IRODORI_PYTHON), str(pipeline.PROJECT_ROOT / "infer.py")])
        self.assertEqual(value("--hf-checkpoint"), "Aratako/Irodori-TTS-v4.1-Small")
        self.assertEqual(value("--ref-wav"), str(self.wavs["speaker"].resolve()))
        self.assertEqual(value("--output-wav"), str(out))
        self.assertEqual(value("--seed"), "7")
        self.assertNotIn("--no-ref", command)

    def test_irodori_command_local_checkpoint_and_no_ref(self):
        options = self.options(tts_text="a", tts_checkpoint=str(self.wavs["reference"]), tts_seed=None)
        command = pipeline.irodori_command(options, Path("r.wav"))
        self.assertIn("--checkpoint", command)
        self.assertNotIn("--hf-checkpoint", command)
        self.assertIn("--no-ref", command)
        self.assertNotIn("--seed", command)

    def test_seedvc_command_defaults_are_singing_settings(self):
        command = pipeline.seedvc_command(self.options(), self.tmp, "ckpt.pth", "cfg.yml")

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(value("--f0-condition"), "True")
        self.assertEqual(value("--auto-f0-adjust"), "False")
        self.assertEqual(value("--semi-tone-shift"), "0")
        self.assertEqual(float(value("--length-adjust")), 1.0)
        self.assertEqual(value("--offline"), "True")

    def test_run_step_failure_raises_with_log_tail(self):
        command = [sys.executable, "-c", "import sys; print('boom detail'); sys.exit(5)"]
        with self.assertRaises(PipelineError) as info:
            pipeline.run_step("failing", command, self.tmp, log=_quiet)
        self.assertIn("exit code 5", str(info.exception))
        self.assertIn("boom detail", str(info.exception))
        log_lines = info.exception.log_path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(log_lines[0].startswith("$ "))
        self.assertIn("boom detail", log_lines[1:])

    def test_run_step_missing_executable(self):
        with self.assertRaisesRegex(PipelineError, "could not start"):
            pipeline.run_step("missing", [str(self.tmp / "nope.exe")], self.tmp, log=_quiet)

    def test_run_pipeline_rejects_before_creating_run_dir(self):
        options = self.options(source_wav=str(self.tmp / "missing.wav"), reference_wav=str(self.tmp / "r.wav"),
                               output_dir=str(self.tmp / "out"))
        with self.assertRaisesRegex(PipelineError, "Invalid request"):
            pipeline.run_pipeline(options, log=_quiet)
        self.assertFalse((self.tmp / "out").exists())

    def test_cli_options_json_and_override(self):
        defaults = self.tmp / "opts.json"
        defaults.write_text(json.dumps({"tts_text": "既定", "semitone_shift": 3, "_comment": "x"}), encoding="utf-8")
        options, dry = cli.parse_options(["--options-json", str(defaults), "--source", str(self.wavs["source"]),
                                          "--semitone-shift", "-2", "--dry-run"])
        self.assertTrue(dry)
        self.assertEqual((options.tts_text, options.semitone_shift), ("既定", -2))
        self.assertTrue(options.offline and options.f0_condition)

    def test_cli_rejects_unknown_json_keys(self):
        defaults = self.tmp / "opts.json"
        defaults.write_text(json.dumps({"semitone": 3}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            cli.parse_options(["--options-json", str(defaults), "--source", str(self.wavs["source"])])

    def test_cli_text_file_utf8_with_bom(self):
        text = self.tmp / "text.txt"
        text.write_text("﻿歌ってみます。\n", encoding="utf-8")
        options, _ = cli.parse_options(["--source", str(self.wavs["source"]), "--tts-text-file", str(text)])
        self.assertEqual(options.tts_text, "歌ってみます。")

    def test_example_options_file_is_valid(self):
        example = pipeline.PROJECT_ROOT / "configs" / "singing" / "example_tts_reference.json"
        options, _ = cli.parse_options(["--options-json", str(example), "--source", str(self.wavs["source"])])
        self.assertEqual(pipeline.validate(options), [])

    def test_runner_config_guard(self):
        self.assertEqual(seedvc_runner.check_config(_config(True), True)["sample_rate"], 44100)
        with self.assertRaisesRegex(seedvc_runner.RunnerError, "not f0-conditioned"):
            seedvc_runner.check_config(_config(False, 22050), True)
        with self.assertRaisesRegex(seedvc_runner.RunnerError, "cannot run"):
            seedvc_runner.check_config(_config(True), False)
        with self.assertRaisesRegex(seedvc_runner.RunnerError, "44.1 kHz"):
            seedvc_runner.check_config(_config(True, 22050), True)

    def test_upstream_singing_preset_passes_guard(self):
        preset = pipeline.SEEDVC_DIR / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
        if not preset.is_file():
            self.skipTest("Seed-VC source not installed")
        import yaml

        facts = seedvc_runner.check_config(yaml.safe_load(preset.read_text(encoding="utf-8")), True)
        self.assertEqual(facts["vocoder"], "nvidia/bigvgan_v2_44khz_128band_512x")

    def test_runner_rejects_unsupported_vocoder(self):
        hifigan = _config(True)
        hifigan["model_params"]["vocoder"] = {"type": "hifigan"}
        wrong_name = _config(True)
        wrong_name["model_params"]["vocoder"]["name"] = "nvidia/bigvgan_v2_22khz_80band_256x"
        for config in (hifigan, wrong_name):
            with self.assertRaisesRegex(seedvc_runner.RunnerError, "must use vocoder"):
                seedvc_runner.check_config(config, True)
        for malformed in ("bigvgan", None, ["bigvgan"], {"name": "nvidia/bigvgan_v2_44khz_128band_512x"}):
            config = _config(True)
            config["model_params"]["vocoder"] = malformed
            with self.subTest(vocoder=malformed), self.assertRaisesRegex(seedvc_runner.RunnerError, "missing"):
                seedvc_runner.check_config(config, True)
        del hifigan["model_params"]["vocoder"]
        with self.assertRaisesRegex(seedvc_runner.RunnerError, "missing"):
            seedvc_runner.check_config(hifigan, True)

    def _runner_files(self):
        import yaml

        root = self.tmp / "runner"
        seedvc = root / "seed-vc"
        seedvc.mkdir(parents=True)
        (seedvc / "inference.py").write_text(FAKE_INFERENCE, encoding="utf-8")
        files = {"source": root / "source.wav", "target": root / "reference.wav", "checkpoint": root / "m.pth",
                 "config": root / "c.yml"}
        for name, path in files.items():
            path.write_bytes(name.encode() * 4)
        files["config"].write_text(yaml.safe_dump(_config(True)), encoding="utf-8")
        snapshot = {name: path.read_bytes() for name, path in files.items()}
        return root, seedvc, files, snapshot

    @staticmethod
    def _runner_args(seedvc, files, output_wav, report_json):
        args = ["--seedvc-dir", str(seedvc), "--output-wav", str(output_wav), "--report-json", str(report_json)]
        for name, path in files.items():
            args += [f"--{name}", str(path)]
        return args

    @staticmethod
    def _run_runner(args):
        cwd, path, argv = os.getcwd(), list(sys.path), list(sys.argv)
        try:
            with mock.patch.dict(sys.modules, {"torch": FAKE_TORCH}), mock.patch.dict(os.environ):
                return seedvc_runner.main(args)
        finally:
            os.chdir(cwd)
            sys.path[:], sys.argv[:] = path, argv

    def test_runner_rejects_output_collisions_before_writing(self):
        root, seedvc, files, snapshot = self._runner_files()
        hardlink = root / "hardlinked reference.wav"
        os.link(files["target"], hardlink)
        cases = {
            "output is source": (files["source"], root / "r.json"),
            "report is config": (root / "o.wav", files["config"]),
            "report is checkpoint": (root / "o.wav", files["checkpoint"]),
            "output is hardlink of target": (hardlink, root / "r.json"),
            "output equals report": (root / "same.out", root / "same.out"),
        }
        for label, (output_wav, report_json) in cases.items():
            with self.subTest(label), self.assertRaises(seedvc_runner.RunnerError):
                self._run_runner(self._runner_args(seedvc, files, output_wav, report_json))
        self.assertEqual({name: path.read_bytes() for name, path in files.items()}, snapshot)
        self.assertEqual(hardlink.read_bytes(), snapshot["target"])
        self.assertFalse(any(root.glob("seedvc_raw*")))
        self.assertFalse((root / "r.json").exists() or (root / "o.wav").exists())

    def test_runner_uses_private_raw_dir_and_keeps_existing_files(self):
        root, seedvc, files, snapshot = self._runner_files()
        old_raw = root / "seedvc_raw"
        old_raw.mkdir()
        (old_raw / "earlier run.wav").write_bytes(b"keep me")
        (root / "unrelated.wav").write_bytes(b"keep me too")
        output_wav, report_json = root / "converted.wav", root / "seedvc_report.json"
        self.assertEqual(self._run_runner(self._runner_args(seedvc, files, output_wav, report_json)), 0)
        self.assertEqual(output_wav.read_bytes(), b"converted")
        report = json.loads(report_json.read_text(encoding="utf-8"))
        raw_arg = Path(report["upstream_argv"][report["upstream_argv"].index("--output") + 1])
        self.assertNotEqual(raw_arg, old_raw)
        self.assertFalse(raw_arg.exists())  # the per-run folder is removed after the move
        self.assertEqual((old_raw / "earlier run.wav").read_bytes(), b"keep me")
        self.assertEqual((root / "unrelated.wav").read_bytes(), b"keep me too")
        self.assertEqual({name: path.read_bytes() for name, path in files.items()}, snapshot)

    def _fake_pipeline_run(self, output_dir, broken_file, broken_text):
        def fake_step(name, command, log_dir, **_kwargs):
            run_dir = log_dir.parent
            outputs = {"prepare_source": {"source.wav": "RIFF"},
                       "seedvc_singing": {"converted.wav": "RIFF", "seedvc_report.json": "{}"},
                       "analyze": {"analysis.json": "{}"}}.get(name, {})
            for file_name, text in outputs.items():
                (run_dir / file_name).write_text(broken_text if file_name == broken_file else text, encoding="utf-8")
            return 0.0

        options = self.options(reference_wav=str(self.wavs["reference"]), output_dir=str(output_dir))
        with mock.patch.object(pipeline, "check_environment", return_value=[]), \
                mock.patch.object(pipeline, "default_seedvc_model", return_value=("c.pth", "c.yml")), \
                mock.patch.object(pipeline, "child_environment", return_value={}), \
                mock.patch.object(pipeline, "git_revision", return_value=None), \
                mock.patch.object(pipeline, "run_step", side_effect=fake_step):
            return pipeline.run_pipeline(options, log=_quiet)

    def test_pipeline_records_unexpected_report_errors(self):
        cases = {"truncated seedvc report": ("seedvc_report.json", '{"elapsed', json.JSONDecodeError),
                 "analysis without converted facts": ("analysis.json", '{"source": {}}', KeyError)}
        for index, (label, (broken_file, text, error)) in enumerate(cases.items()):
            output_dir = self.tmp / f"out{index}"
            with self.subTest(label):
                with self.assertRaises(error):
                    self._fake_pipeline_run(output_dir, broken_file, text)
                (run_dir,) = output_dir.iterdir()
                metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "failed")
                self.assertTrue(metadata["error"].startswith(error.__name__))
                self.assertIn("finished_at", metadata)
                self.assertIn("(failed)", (run_dir / "summary_ja.md").read_text(encoding="utf-8"))

    def test_cli_keeps_json_offline_and_f0_report_unless_flag_given(self):
        defaults = self.tmp / "opts.json"
        defaults.write_text(json.dumps({"tts_text": "a", "offline": False, "f0_report": False}), encoding="utf-8")
        base = ["--options-json", str(defaults), "--source", str(self.wavs["source"])]
        options, _ = cli.parse_options(base)
        self.assertEqual((options.offline, options.f0_report), (False, False))
        defaults.write_text(json.dumps({"tts_text": "a", "offline": True, "f0_report": True}), encoding="utf-8")
        options, _ = cli.parse_options(base + ["--online", "--no-f0-report"])
        self.assertEqual((options.offline, options.f0_report), (False, False))

    def test_f0_compare_matches_requested_shift(self):
        src = np.concatenate([np.linspace(220.0, 440.0, 50), [np.nan] * 5])
        result = audio_tool.compare_contours(src, src * 2 ** (2 / 12), 2)
        self.assertEqual(result["median_abs_cents_error"], 0.0)
        self.assertEqual(result["within_50_cents_ratio"], 1.0)
        self.assertEqual(result["f0_correlation"], 1.0)
        self.assertNotIn("not_comparable_reason", result)

    def test_f0_compare_skips_when_timing_changes(self):
        src = np.linspace(220.0, 440.0, 50)
        result = audio_tool.compare_contours(src, src, 0, length_adjust=1.5)
        self.assertIn("length_adjust=1.5", result["not_comparable_reason"])
        self.assertFalse({"median_abs_cents_error", "f0_correlation", "frames_compared"} & set(result))

    def test_f0_compare_auto_adjust_drops_fixed_shift_metrics(self):
        src = np.linspace(220.0, 440.0, 50)
        result = audio_tool.compare_contours(src, src * 1.3, 3, auto_f0_adjust=True)
        self.assertIsNone(result["expected_shift_semitones"])
        self.assertIn("auto_f0_adjust", result["cents_not_comparable_reason"])
        self.assertNotIn("median_abs_cents_error", result)
        self.assertEqual(result["f0_correlation"], 1.0)

    def test_f0_compare_constant_contour_is_valid_json(self):
        flat = np.full(40, 330.0)
        result = audio_tool.compare_contours(flat, flat, 0)
        self.assertIsNone(result["f0_correlation"])
        self.assertEqual(result["median_abs_cents_error"], 0.0)
        json.dumps(result, allow_nan=False)
        unvoiced = audio_tool.compare_contours(np.full(40, np.nan), flat, 0)
        self.assertIn("fewer than 10", unvoiced["not_comparable_reason"])
        json.dumps(unvoiced, allow_nan=False)
        self.assertTrue(math.isclose(audio_tool.compare_contours(flat, flat * 2, 0)["octave_error_ratio"], 1.0))

    def test_runner_rejects_missing_inputs(self):
        args = ["--seedvc-dir", str(self.tmp), "--source", str(self.tmp / "s.wav"),
                "--target", str(self.tmp / "t.wav"), "--output-wav", str(self.tmp / "o.wav"),
                "--report-json", str(self.tmp / "r.json"), "--checkpoint", str(self.tmp / "c.pth"),
                "--config", str(self.tmp / "c.yml")]
        with self.assertRaisesRegex(seedvc_runner.RunnerError, "source not found"):
            seedvc_runner.main(args)


if __name__ == "__main__":
    unittest.main()
