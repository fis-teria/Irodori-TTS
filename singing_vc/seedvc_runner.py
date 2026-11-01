"""Run upstream Seed-VC v1 ``inference.py`` for singing conversion, inside the Seed-VC venv.

The checkpoint and config are always passed explicitly, so upstream never
auto-downloads or substitutes its speech (non-f0) model. The config is checked
before any model loads: f0 conditioning must match the request and the model
must be the 44.1 kHz BigVGAN variant. Upstream writes into a fresh per-run
temporary folder, so existing files are never deleted. Only the standard
library, PyYAML and the Seed-VC venv's torch are used.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
from pathlib import Path


SINGING_SAMPLE_RATE = 44100
SINGING_VOCODER = {"type": "bigvgan", "name": "nvidia/bigvgan_v2_44khz_128band_512x"}


class RunnerError(RuntimeError):
    pass


def str2bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "t"}:
        return True
    if lowered in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def check_config(config: dict, f0_condition: bool) -> dict:
    """Return model facts from a Seed-VC config, rejecting mismatched requests."""
    try:
        sample_rate = int(config["preprocess_params"]["sr"])
        params = config["model_params"]
        config_f0 = bool(params["length_regulator"]["f0_condition"]) and bool(params["DiT"]["f0_condition"])
        vocoder = params["vocoder"]
        vocoder_type, vocoder_name = vocoder["type"], vocoder.get("name")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RunnerError(f"Seed-VC config is missing required keys: {exc!r}") from exc
    if f0_condition and not config_f0:
        raise RunnerError("f0-condition=true was requested, but the config is not f0-conditioned.")
    if not f0_condition and config_f0:
        raise RunnerError(
            "f0-condition=false cannot run the f0-conditioned singing model "
            "(upstream would switch to 22.05 kHz speech settings); pass a non-f0 config explicitly."
        )
    if f0_condition and sample_rate != SINGING_SAMPLE_RATE:
        raise RunnerError(f"Singing model must be 44.1 kHz, config says {sample_rate} Hz.")
    if f0_condition and (vocoder_type, vocoder_name) != (SINGING_VOCODER["type"], SINGING_VOCODER["name"]):
        raise RunnerError(
            f"Singing model must use vocoder {SINGING_VOCODER['type']} {SINGING_VOCODER['name']}, "
            f"config says {vocoder_type!r} {vocoder_name!r}."
        )
    return {
        "sample_rate": sample_rate,
        "f0_condition": config_f0,
        "vocoder": vocoder_name or vocoder_type,
        "speech_tokenizer": params.get("speech_tokenizer", {}).get("name"),
    }


def _same_file(a: Path, b: Path) -> bool:
    if os.path.normcase(str(a)) == os.path.normcase(str(b)):
        return True
    try:
        return os.path.samefile(a, b)  # also catches hardlinks and symlinks to the same file
    except OSError:
        return False  # at least one does not exist yet, so they cannot be the same file


def check_output_paths(args: argparse.Namespace) -> None:
    """Reject outputs that would overwrite each other or any input; call before writing anything."""
    if args.output_wav.is_dir() or args.report_json.is_dir():
        raise RunnerError("output-wav and report-json must be file paths, not folders.")
    if _same_file(args.output_wav, args.report_json):
        raise RunnerError(f"output-wav and report-json are the same file: {args.output_wav}")
    for out_name in ("output_wav", "report_json"):
        for in_name in ("source", "target", "checkpoint", "config"):
            if _same_file(getattr(args, out_name), getattr(args, in_name)):
                raise RunnerError(
                    f"{out_name.replace('_', '-')} would overwrite {in_name}: {getattr(args, in_name)}"
                )


def upstream_argv(args: argparse.Namespace, raw_dir: Path) -> list[str]:
    return [
        "inference.py",
        "--source", str(args.source),
        "--target", str(args.target),
        "--output", str(raw_dir),
        "--checkpoint", str(args.checkpoint),
        "--config", str(args.config),
        "--diffusion-steps", str(args.diffusion_steps),
        "--length-adjust", repr(float(args.length_adjust)),
        "--inference-cfg-rate", repr(float(args.inference_cfg_rate)),
        "--f0-condition", str(bool(args.f0_condition)),
        "--auto-f0-adjust", str(bool(args.auto_f0_adjust)),
        "--semi-tone-shift", str(int(args.semi_tone_shift)),
        "--fp16", str(bool(args.fp16)),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seedvc-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", type=str2bool, default=True)
    parser.add_argument("--auto-f0-adjust", type=str2bool, default=False)
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--offline", type=str2bool, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Resolve everything before upstream needs os.chdir(seedvc_dir).
    args.output_wav = args.output_wav.resolve()
    args.report_json = args.report_json.resolve()
    for name in ("seedvc_dir", "source", "target", "checkpoint", "config"):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        exists = path.is_dir() if name == "seedvc_dir" else path.is_file()
        if not exists:
            raise RunnerError(f"{name.replace('_', '-')} not found: {path}")
    if not (args.seedvc_dir / "inference.py").is_file():
        raise RunnerError(f"Seed-VC inference.py not found in {args.seedvc_dir}")
    if not 0.0 < args.length_adjust <= 4.0:
        raise RunnerError("length-adjust must be in (0, 4].")
    if args.diffusion_steps < 1:
        raise RunnerError("diffusion-steps must be >= 1.")
    check_output_paths(args)

    import yaml

    with args.config.open(encoding="utf-8") as stream:
        model_facts = check_config(yaml.safe_load(stream), bool(args.f0_condition))

    # Must be set before upstream imports huggingface_hub; models were fetched by setup.
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    output_wav = args.output_wav
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    # A fresh folder owned by this run: nothing that existed before is touched or deleted.
    raw_dir = Path(tempfile.mkdtemp(prefix="seedvc_raw_", dir=output_wav.parent))
    try:
        os.chdir(args.seedvc_dir)  # upstream resolves ./checkpoints and ./modules from cwd
        sys.path.insert(0, str(args.seedvc_dir))
        sys.argv = upstream_argv(args, raw_dir)
        started = time.perf_counter()
        runpy.run_path(str(args.seedvc_dir / "inference.py"), run_name="__main__")
        elapsed = time.perf_counter() - started

        produced = sorted(raw_dir.glob("*.wav"))
        if len(produced) != 1:
            raise RunnerError(f"Expected one Seed-VC output in {raw_dir}, found {len(produced)}.")
        produced[0].replace(output_wav)
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)

    import torch

    cuda = torch.cuda.is_available()
    report = {
        "upstream_argv": sys.argv,
        "model": model_facts,
        "offline": bool(args.offline),
        "elapsed_seconds": round(elapsed, 3),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "cuda_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "output_wav": str(output_wav),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[seedvc] saved {output_wav}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"[seedvc] error: {exc}", file=sys.stderr)
        raise SystemExit(2)
