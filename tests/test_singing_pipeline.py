"""Argument, command and failure-path checks for the singing prototype (no models loaded)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from singing_vc import cli, pipeline, seedvc_runner
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

    def test_runner_rejects_missing_inputs(self):
        args = ["--seedvc-dir", str(self.tmp), "--source", str(self.tmp / "s.wav"),
                "--target", str(self.tmp / "t.wav"), "--output-wav", str(self.tmp / "o.wav"),
                "--report-json", str(self.tmp / "r.json"), "--checkpoint", str(self.tmp / "c.pth"),
                "--config", str(self.tmp / "c.yml")]
        with self.assertRaisesRegex(seedvc_runner.RunnerError, "source not found"):
            seedvc_runner.main(args)


if __name__ == "__main__":
    unittest.main()
