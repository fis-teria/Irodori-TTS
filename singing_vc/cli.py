"""Convert an isolated vocal to a target voice: Irodori reference speech + Seed-VC v1 singing model.

Examples (PowerShell):
  .\\run_singing.ps1 --source "D:\\vocals\\song a.wav" --tts-text "こんにちは。" --tts-caption "落ち着いた女性の声"
  .\\run_singing.ps1 --source vocal.wav --reference-wav target_voice.wav --semitone-shift -2
  .\\run_singing.ps1 --options-json configs\\singing\\example_tts_reference.json --source vocal.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singing_vc.pipeline import PipelineError, SingingOptions, run_pipeline, validate


def bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--options-json", type=Path, help="JSON file of option defaults (keys = SingingOptions fields).")
    parser.add_argument("--source", dest="source_wav", help="Isolated source vocal WAV (no accompaniment).")
    parser.add_argument("--source-start", type=float, default=0.0, help="Seconds to skip in the source.")
    parser.add_argument("--source-duration", type=float, help="Seconds of source to convert (default: all).")

    ref = parser.add_argument_group("target reference (choose one)")
    ref.add_argument("--reference-wav", help="Existing target-voice WAV used directly as Seed-VC reference.")
    ref.add_argument("--tts-text", help="Generate the reference with Irodori-TTS from this text.")
    ref.add_argument("--tts-text-file", type=Path, help="UTF-8 file containing --tts-text (avoids console encoding issues).")

    tts = parser.add_argument_group("Irodori-TTS (reference generation)")
    tts.add_argument("--tts-caption", help="Caption / voice-design text.")
    tts.add_argument("--tts-caption-file", type=Path, help="UTF-8 file containing --tts-caption.")
    tts.add_argument("--tts-checkpoint", help="Local checkpoint (.safetensors/.pt); overrides --tts-hf-checkpoint.")
    tts.add_argument("--tts-hf-checkpoint", default=SingingOptions.tts_hf_checkpoint)
    tts.add_argument("--tts-lora-adapter", help="PEFT LoRA adapter directory.")
    tts.add_argument("--tts-speaker-wav", help="Irodori speaker reference WAV (otherwise --no-ref).")
    tts.add_argument("--tts-seed", type=int, default=SingingOptions.tts_seed)
    tts.add_argument("--tts-num-steps", type=int)
    tts.add_argument("--tts-seconds", type=float)
    tts.add_argument("--tts-extra-arg", dest="tts_extra_args", action="append", default=[],
                     help="Extra raw infer.py argument; repeat per token, e.g. --tts-extra-arg=--cfg-scale-text --tts-extra-arg=4.")

    vc = parser.add_argument_group("Seed-VC singing conversion")
    vc.add_argument("--diffusion-steps", type=int, default=SingingOptions.diffusion_steps)
    vc.add_argument("--length-adjust", type=float, default=SingingOptions.length_adjust)
    vc.add_argument("--inference-cfg-rate", type=float, default=SingingOptions.inference_cfg_rate)
    vc.add_argument("--f0-condition", type=bool_arg, default=True)
    vc.add_argument("--auto-f0-adjust", type=bool_arg, default=False)
    vc.add_argument("--semitone-shift", type=int, default=0)
    vc.add_argument("--fp16", type=bool_arg, default=True)
    vc.add_argument("--seedvc-checkpoint", help="Override DiT checkpoint (default: pinned v1 singing ft_ema).")
    vc.add_argument("--seedvc-config", help="Override Seed-VC config (default: upstream f0 44k preset).")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", default=SingingOptions.output_dir)
    out.add_argument("--run-name", help="Label appended to the timestamped run folder.")
    out.add_argument("--online", action="store_true", help="Allow Hugging Face network access (default: offline cache).")
    out.add_argument("--no-f0-report", action="store_true", help="Skip the pYIN F0 comparison.")
    out.add_argument("--dry-run", action="store_true", help="Validate and print options without running models.")
    return parser


def parse_options(argv: list[str] | None = None) -> tuple[SingingOptions, bool]:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    names = {f.name for f in fields(SingingOptions)}
    if pre.options_json is not None:
        try:
            defaults = json.loads(pre.options_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --options-json: {exc}")
        unknown = sorted(set(defaults) - names - {"_comment"})
        if unknown:
            parser.error(f"unknown keys in {pre.options_json}: {', '.join(unknown)}")
        defaults.pop("_comment", None)
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in ("tts_text", "tts_caption"):
        file_value = getattr(args, f"{name}_file")
        if file_value is not None:
            if getattr(args, name):
                parser.error(f"use either --{name.replace('_', '-')} or its -file variant")
            setattr(args, name, file_value.read_text(encoding="utf-8-sig").strip())
    if not args.source_wav:
        parser.error("--source is required")
    values = {name: getattr(args, name) for name in names if hasattr(args, name)}
    values["offline"] = not args.online
    values["f0_report"] = not args.no_f0_report
    return SingingOptions(**values), bool(args.dry_run)


def main(argv: list[str] | None = None) -> int:
    options, dry_run = parse_options(argv)
    if dry_run:
        problems = validate(options)
        print(json.dumps({"options": options.__dict__, "problems": problems}, indent=2, ensure_ascii=False))
        return 1 if problems else 0
    try:
        metadata = run_pipeline(options)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"metadata: {Path(metadata['run_dir']) / 'metadata.json'}")
    return 0 if metadata["status"] == "succeeded" else 3


if __name__ == "__main__":
    raise SystemExit(main())
