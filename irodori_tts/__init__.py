"""Irodori-TTS package: text-conditioned RF diffusion over DACVAE latents."""

from .local_env import configure_project_environment

configure_project_environment()

from .config import ModelConfig, TrainConfig
from .lora import LORA_TARGET_PRESETS
from .model import TextToLatentRFDiT
from .tokenizer import PretrainedTextTokenizer

__all__ = [
    "LORA_TARGET_PRESETS",
    "ModelConfig",
    "PretrainedTextTokenizer",
    "TextToLatentRFDiT",
    "TrainConfig",
]
