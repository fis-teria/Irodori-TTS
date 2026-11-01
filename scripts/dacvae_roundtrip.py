"""Encode a WAV with Irodori's DACVAE codec and decode it back (no TTS model involved).

A first gate for native Irodori singing: if the codec latent cannot carry a
sung vocal, a TTS that predicts those latents cannot sing either. Passing it
does not show that a TTS would learn to sing. Run inside the Irodori ``.venv``.

  .venv\\Scripts\\python.exe scripts\\dacvae_roundtrip.py --input in.wav --output out.wav --json out.json

``--input-level model-card`` (opt-in) encodes at the model card's -16 LUFS and divides the
same gain back out after decoding, so the output stays at the input level. On quiet inputs it
lowers the fixed decoder floor in pauses; it does not change the sung-harmonic losses
(docs/CODEC_NOISE_DIAGNOSIS_JA.md). The default stays the raw level for comparability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _same_file(a: Path, b: Path) -> bool:
    if os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b)):
        return True
    try:
        return os.path.samefile(a, b)  # hardlinks / symlinks
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, nargs="+", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-level", choices=("raw", "model-card"), default="raw",
                        help="model-card: encode at -16 LUFS, then restore the input level after decoding")
    args = parser.parse_args(argv)
    if len(args.input) != len(args.output):
        parser.error("--input and --output need the same number of paths")
    outputs = [*args.output, args.json]
    for i, out in enumerate(outputs):
        if any(_same_file(out, other) for other in outputs[i + 1:]) or any(_same_file(out, s) for s in args.input):
            parser.error(f"output path collides with another output or an input: {out}")

    from irodori_tts.local_env import configure_project_environment

    configure_project_environment()
    import soundfile as sf
    import torch
    import torchaudio

    from irodori_tts.codec import DACVAECodec
    from singing_vc import codec_diag

    # Level preserved (no loudness normalisation) so the decode can be compared sample for sample.
    codec = DACVAECodec.load(repo_id=args.codec_repo, device=args.device, normalize_db=None)
    items = []
    for source, target in zip(args.input, args.output):
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
        waveform, rate = torch.from_numpy(audio.T.copy()), sample_rate
        started = time.perf_counter()
        gain_db = 0.0
        if args.input_level == "model-card":  # the wrapper's mono mix + resample, then -16 LUFS
            waveform = waveform.mean(dim=0, keepdim=True)
            if rate != codec.sample_rate:
                waveform = torchaudio.functional.resample(waveform, rate, codec.sample_rate)
            rate = codec.sample_rate
            normalized = codec._normalize_loudness(waveform[0], rate, -16.0)  # pure scalar gain
            gain_db = codec_diag.scalar_gain_db(waveform[0].numpy(), normalized.numpy())
            waveform = normalized[None]
        latent = codec.encode_waveform(waveform, rate, normalize_db=None, ensure_max=True)
        decoded = codec.decode_latent(latent)[0, 0].float().cpu().numpy()
        if gain_db:
            decoded = codec_diag.restore_gain(decoded, gain_db)
        elapsed = time.perf_counter() - started
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), decoded, codec.sample_rate, subtype="FLOAT")
        items.append({"input": str(source), "output": str(target), "input_sample_rate": sample_rate,
                      "codec_sample_rate": codec.sample_rate, "conditioning_gain_db": round(gain_db, 4),
                      "latent_shape": list(latent.shape),
                      "latent_frames_per_second": round(latent.shape[1] / (audio.shape[0] / sample_rate), 3),
                      "seconds": round(elapsed, 3)})
        print(f"[dacvae] {source.name} -> {target.name} latent {tuple(latent.shape)}", flush=True)
    report = {"codec_repo": args.codec_repo, "normalize_db": None, "input_level": args.input_level,
              "deterministic_encode": True, "deterministic_decode": True,
              "watermark_branch": "skipped by DACVAECodec.load", "items": items}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
