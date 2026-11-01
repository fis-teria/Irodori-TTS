"""Encode a WAV with Irodori's DACVAE codec and decode it back (no TTS model involved).

A first gate for native Irodori singing: if the codec latent cannot carry a
sung vocal, a TTS that predicts those latents cannot sing either. Passing it
does not show that a TTS would learn to sing. Run inside the Irodori ``.venv``.

  .venv\\Scripts\\python.exe scripts\\dacvae_roundtrip.py --input in.wav --output out.wav --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, nargs="+", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if len(args.input) != len(args.output):
        parser.error("--input and --output need the same number of paths")

    from irodori_tts.local_env import configure_project_environment

    configure_project_environment()
    import soundfile as sf
    import torch

    from irodori_tts.codec import DACVAECodec

    # Level preserved (no loudness normalisation) so the decode can be compared sample for sample.
    codec = DACVAECodec.load(repo_id=args.codec_repo, device=args.device, normalize_db=None)
    items = []
    for source, target in zip(args.input, args.output):
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T.copy())
        started = time.perf_counter()
        latent = codec.encode_waveform(waveform, sample_rate, normalize_db=None, ensure_max=True)
        decoded = codec.decode_latent(latent)[0, 0].float().cpu().numpy()
        elapsed = time.perf_counter() - started
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), decoded, codec.sample_rate, subtype="FLOAT")
        items.append({"input": str(source), "output": str(target), "input_sample_rate": sample_rate,
                      "codec_sample_rate": codec.sample_rate, "latent_shape": list(latent.shape),
                      "latent_frames_per_second": round(latent.shape[1] / (audio.shape[0] / sample_rate), 3),
                      "seconds": round(elapsed, 3)})
        print(f"[dacvae] {source.name} -> {target.name} latent {tuple(latent.shape)}", flush=True)
    report = {"codec_repo": args.codec_repo, "normalize_db": None, "deterministic_encode": True,
              "deterministic_decode": True, "watermark_branch": "skipped by DACVAECodec.load", "items": items}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
