"""Reuse Hugging Face caches from another Irodori workspace without modifying it.

Files are hardlinked when source and destination share a volume (no extra disk
space), otherwise copied. The source workspace is only read. Hugging Face never
rewrites cached files in place, so the shared inodes stay unchanged; deleting
the prototype copy does not affect the source. Standard library only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRODORI_HUB = PROJECT_ROOT / "model_data" / "huggingface" / "hub"
SEEDVC_HUB = PROJECT_ROOT / ".singing" / "seed-vc" / "checkpoints" / "hf_cache"

# Repositories that Irodori v4.1-Small inference reads (model, codec, watermark, tokenizer).
IRODORI_REPOS = (
    "Aratako/Irodori-TTS-v4.1-Small",
    "Aratako/Semantic-DACVAE-Japanese-32dim",
    "sony/silentcipher",
    "sbintuitions/modernbert-ja-310m",
)
# Seed-VC's content encoder is the same public whisper-small repository.
SEEDVC_REPOS = ("openai/whisper-small",)


def repo_folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def link_tree(source: Path, destination: Path) -> tuple[int, int]:
    """Mirror files from source into destination; return (linked, copied)."""
    linked = copied = 0
    for root, _dirs, files in os.walk(source):
        target_dir = destination / Path(root).relative_to(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = Path(root) / name
            dst = target_dir / name
            if dst.exists():
                continue
            if name.endswith(".lock") or src.stat().st_size < 1 << 20:
                # Small metadata files are copied so no mutable file is shared.
                shutil.copy2(src, dst)
                copied += 1
                continue
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
    return linked, copied


def ensure_main_ref(repo_dir: Path) -> None:
    """Offline loads resolve ``main`` through refs/main; add it if the source lacks it."""
    ref = repo_dir / "refs" / "main"
    snapshots = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").is_dir() else []
    if ref.exists() or len(snapshots) != 1:
        return
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(snapshots[0].name, encoding="ascii")


def seed(source_workspace: Path, repos: dict[Path, tuple[str, ...]]) -> list[str]:
    source_hub = source_workspace / "model_data" / "huggingface" / "hub"
    if not source_hub.is_dir():
        raise FileNotFoundError(f"No Hugging Face cache in source workspace: {source_hub}")
    if source_workspace.resolve() == PROJECT_ROOT.resolve():
        raise ValueError("Source workspace must differ from this checkout.")
    report = []
    for destination_hub, repo_ids in repos.items():
        for repo_id in repo_ids:
            src = source_hub / repo_folder(repo_id)
            if not src.is_dir():
                report.append(f"skip {repo_id}: not cached in source")
                continue
            dst = destination_hub / repo_folder(repo_id)
            linked, copied = link_tree(src, dst)
            ensure_main_ref(dst)
            report.append(f"{repo_id}: hardlinked={linked} copied={copied} -> {dst}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_workspace", type=Path, help="Existing Irodori-TTS_V4 checkout to read from.")
    parser.add_argument("--skip-seedvc", action="store_true", help="Do not seed Seed-VC's whisper cache.")
    args = parser.parse_args(argv)
    repos: dict[Path, tuple[str, ...]] = {IRODORI_HUB: IRODORI_REPOS}
    if not args.skip_seedvc:
        repos[SEEDVC_HUB] = SEEDVC_REPOS
    try:
        for line in seed(args.source_workspace.expanduser().resolve(), repos):
            print(line, flush=True)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
