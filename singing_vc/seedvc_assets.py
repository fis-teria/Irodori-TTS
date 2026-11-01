"""Fetch the Seed-VC v1 singing assets at pinned revisions, inside the Seed-VC venv.

Files land in the cache layout upstream ``inference.py`` reads from its working
directory (``checkpoints`` for hf_utils, ``checkpoints/hf_cache`` for BigVGAN and
Whisper). ``refs/main`` is pointed at the pinned commit so offline runs
(HF_HUB_OFFLINE=1) resolve exactly these revisions. Re-running is cheap: cached
files are reused. ``--local-files-only`` verifies the cache without network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# (cache, repo_id, revision, files). "hf_utils" = <seed-vc>/checkpoints, "hub" = <seed-vc>/checkpoints/hf_cache.
ASSETS = (
    ("hf_utils", "Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e",
     ("DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
      "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")),
    ("hf_utils", "lj1995/VoiceConversionWebUI", "e6d0c1a17da07c33557852f9dfa2bd44cc75737d", ("rmvpe.pt",)),
    ("hf_utils", "funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede", ("campplus_cn_common.bin",)),
    ("hub", "nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb",
     ("config.json", "bigvgan_generator.pt")),
    ("hub", "openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d",
     ("config.json", "model.safetensors", "preprocessor_config.json")),
)
SINGING_CHECKPOINT = ASSETS[0][3][0]
SINGING_CONFIG = "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_main(cache_dir: Path, repo_id: str, revision: str) -> None:
    ref = cache_dir / ("models--" + repo_id.replace("/", "--")) / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision, encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seedvc-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    from huggingface_hub import hf_hub_download

    seedvc_dir = args.seedvc_dir.resolve()
    caches = {"hf_utils": seedvc_dir / "checkpoints", "hub": seedvc_dir / "checkpoints" / "hf_cache"}
    records = []
    for cache, repo_id, revision, files in ASSETS:
        for filename in files:
            path = Path(hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision,
                cache_dir=caches[cache], local_files_only=args.local_files_only,
            ))
            records.append({"repo_id": repo_id, "revision": revision, "file": filename,
                            "bytes": path.stat().st_size, "path": str(path)})
            print(f"{repo_id}@{revision[:8]} {filename} ({path.stat().st_size} bytes)", flush=True)
        pin_main(caches[cache], repo_id, revision)

    checkpoint = next(Path(r["path"]) for r in records if r["file"] == SINGING_CHECKPOINT)
    config = seedvc_dir / SINGING_CONFIG
    if not config.is_file():
        print(f"error: missing upstream config {config}", file=sys.stderr)
        return 2
    import yaml

    hf_config = next(Path(r["path"]) for r in records if r["file"].endswith(".yml"))
    # The two copies differ only in training fields (log_dir, pretrained_model).
    model_keys = ("preprocess_params", "model_params")
    repo_yaml, hf_yaml = (yaml.safe_load(p.read_text(encoding="utf-8")) for p in (config, hf_config))
    manifest = {
        "singing_checkpoint": str(checkpoint),
        "singing_checkpoint_sha256": sha256(checkpoint),
        "singing_config": str(config),
        "singing_config_model_params_match_hf_copy": all(repo_yaml[k] == hf_yaml[k] for k in model_keys),
        "assets": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
