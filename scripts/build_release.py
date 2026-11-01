"""Build a Windows source/setup ZIP without local models, credentials or environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    '.gitignore', '.python-version', 'LICENSE', 'README.md', 'pyproject.toml',
    'requirements.txt', 'uv.lock', 'infer.py', 'train.py', 'prepare_manifest.py',
    'convert_checkpoint_to_safetensors.py', 'quantize_checkpoint.py',
    'gradio_app.py', 'gradio_app_voicedesign.py', 'setup.bat', 'setup_windows.ps1',
    'run.bat', 'run_python.bat', 'run_python.ps1',
    'run_webui_reference_audio.bat', 'run_webui_voice_design.bat',
)
SOURCE_DIRS = {
    'irodori_tts': {'.py'}, 'configs': {'.yaml', '.yml'},
    'docs': {'.md'}, 'scripts': {'.py', '.ps1'}, 'tests': {'.py'},
}
EXCLUDED_PARTS = {'.git', '.venv', '.cache', '.tools', '__pycache__', 'model_data', 'outputs', 'data'}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_payload() -> dict[str, bytes]:
    payload = {name: (ROOT / name).read_bytes() for name in ROOT_FILES}
    for folder, extensions in SOURCE_DIRS.items():
        for path in sorted((ROOT / folder).rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.suffix not in extensions:
                continue
            if path.is_symlink() or any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if relative.as_posix() == 'docs/WINDOWS_DISTRIBUTION.md':
                continue
            payload[relative.as_posix()] = path.read_bytes()
    guide = ROOT / 'docs/WINDOWS_DISTRIBUTION.md'
    if not guide.is_file():
        guide = ROOT / 'LOCAL_SETUP_WINDOWS.md'
    payload['LOCAL_SETUP_WINDOWS.md'] = guide.read_bytes()
    introduction = (
        'Irodori-TTS V4 / V4.1 Windows セットアップ\r\n\r\n'
        '1. ZIPをすべて展開してください。\r\n'
        '2. Python 3.10以上とGit for Windowsを用意し、PATHで実行できるようにしてください。\r\n'
        '3. setup.batを実行し、Small・Large・Animeからダウンロードするモデルを選びます。\r\n'
        '4. セットアップ完了後、run.batからWebUIを起動します。\r\n\r\n'
        'モデル・仮想環境・キャッシュは展開したフォルダ内に作成されます。\r\n'
        '初回セットアップにはインターネット接続が必要です。\r\n'
        '詳しい手順はLOCAL_SETUP_WINDOWS.md、コードのライセンスはLICENSEを参照してください。\r\n'
    )
    payload['はじめに.txt'] = introduction.encode('utf-8-sig')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-id', required=True, help='For example: 20261024-r1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    if not re.fullmatch(r'\d{8}-r\d+', args.release_id):
        parser.error('--release-id must use YYYYMMDD-rN')
    payload = collect_payload()
    package_name = f'Irodori-TTS_V4_Windows_{args.release_id}'
    archive_path = args.output_dir / f'{package_name}.zip'
    if archive_path.exists():
        raise FileExistsError(f'Release already exists; choose a new release-id: {archive_path}')
    commit = subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=False,
    )
    manifest = {
        'package': package_name,
        'upstream_repository': 'https://github.com/Aratako/Irodori-TTS',
        'upstream_base_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'edition': 'Windows setup with local modifications',
        'contains_model_weights': False,
        'files': {
            name: {'bytes': len(data), 'sha256': sha256(data)}
            for name, data in sorted(payload.items())
        },
    }
    payload['RELEASE_MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    date = args.release_id[:8]
    timestamp = (int(date[:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
    temporary = archive_path.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(f'Irodori-TTS_V4/{name}', date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for name, metadata in manifest['files'].items():
            assert sha256(archive.read(f'Irodori-TTS_V4/{name}')) == metadata['sha256'], name
    temporary.rename(archive_path)
    digest = sha256(archive_path.read_bytes())
    archive_path.with_suffix('.zip.sha256').write_text(
        f'{digest}  {archive_path.name}\n', encoding='ascii'
    )
    print(json.dumps({'archive': str(archive_path), 'bytes': archive_path.stat().st_size,
                      'files': len(payload), 'sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
