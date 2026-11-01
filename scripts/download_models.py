"""Download V4 runtime assets into this checkout, recording exact revisions."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

# Configure paths before importing dependencies (including huggingface_hub constants).
environment = runpy.run_path(str(Path(__file__).resolve().parents[1] / "irodori_tts/local_env.py"))
environment['configure_project_environment']()
MODEL_DATA = environment['MODEL_DATA']

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-repo', action='append',
        help='Model repository to download; repeat for multiple models. Defaults to v4.1-Small.',
    )
    parser.add_argument('--skip-silentcipher', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    model_repos = list(dict.fromkeys(args.model_repo or ['Aratako/Irodori-TTS-v4.1-Small']))
    specifications = [
        (repo, ['model.safetensors', 'tokenizer/*', 'README.md']) for repo in model_repos
    ]
    specifications.append(
        ('Aratako/Semantic-DACVAE-Japanese-32dim', ['weights.pth', 'README.md'])
    )
    if not args.skip_silentcipher:
        # SilentCipher asks HF for a complete snapshot, even for 44.1 kHz only.
        specifications.append(('sony/silentcipher', None))
    record_path = MODEL_DATA / 'installed_models.json'
    records = {
        item['repo_id']: item
        for item in (json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else [])
    }
    for repo, patterns in specifications:
        folder = Path(snapshot_download(
            repo_id=repo, allow_patterns=patterns,
            local_files_only=args.local_files_only,
        ))
        expected = 'weights.pth' if 'Semantic-DACVAE' in repo else 'model.safetensors'
        if repo == 'sony/silentcipher':
            expected = '44_1_khz/73999_iteration/hparams.yaml'
        if not (folder / expected).is_file():
            raise FileNotFoundError(f'Incomplete model download: {folder / expected}')
        records[repo] = {'repo_id': repo, 'revision': folder.name, 'path': str(folder)}
        print(f'{repo} @ {folder.name} -> {folder}', flush=True)
    record_path.write_text(
        json.dumps(list(records.values()), indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
