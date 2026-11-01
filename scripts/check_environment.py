"""Check actual cache locations, dependencies and CUDA without loading TTS weights."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import irodori_tts
from irodori_tts.local_env import PROJECT_ROOT, project_paths

import gradio
import huggingface_hub.constants as hf_constants
import peft
import torch
import torchaudio
import torchcodec
from datasets import config as dataset_config


def main() -> None:
    actual = {
        'HF_HOME': hf_constants.HF_HOME,
        'HF_HUB_CACHE': hf_constants.HF_HUB_CACHE,
        'HF_DATASETS_CACHE': str(dataset_config.HF_DATASETS_CACHE),
        'TORCH_HOME': str(Path(torch.hub.get_dir()).parent),
        'GRADIO_TEMP_DIR': gradio.utils.get_upload_folder(),
    }
    for name, expected in project_paths().items():
        assert Path(os.environ[name]).resolve() == expected.resolve(), (name, os.environ[name])
    for name, value in actual.items():
        assert Path(value).resolve() == project_paths()[name].resolve(), (name, value)
    assert Path(sys.prefix).resolve() == (PROJECT_ROOT / '.venv').resolve(), sys.prefix
    packages = {
        name: importlib.metadata.version(name)
        for name in ('torch', 'torchaudio', 'torchcodec', 'transformers', 'peft', 'gradio', 'dacvae', 'safetensors')
    }
    gpu = None
    if torch.version.cuda:
        assert torch.cuda.is_available(), 'CUDA build installed but the GPU is unavailable'
        value = (torch.ones(4, device='cuda') * 2).sum().item()
        assert value == 8
        gpu = torch.cuda.get_device_name(0)
    print(json.dumps({'python': sys.executable, 'packages': packages, 'cuda': torch.version.cuda,
                      'ffmpeg': shutil.which('ffmpeg'),
                      'gpu': gpu, 'paths': actual}, indent=2))


if __name__ == '__main__':
    main()
