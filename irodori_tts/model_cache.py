from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR_ENV = "IRODORI_TTS_MODEL_DIR"
OUTPUT_DIR_ENV = "IRODORI_TTS_OUTPUT_DIR"
DATA_DIR_ENV = "IRODORI_TTS_DATA_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATA_DIR = PROJECT_ROOT / "model_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _resolve_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


def model_data_dir() -> Path:
    raw = os.environ.get(MODEL_DIR_ENV)
    if raw is None or str(raw).strip() == "":
        return DEFAULT_MODEL_DATA_DIR
    return _resolve_path(raw)


def output_dir() -> Path:
    raw = os.environ.get(OUTPUT_DIR_ENV)
    if raw is None or str(raw).strip() == "":
        return DEFAULT_OUTPUT_DIR
    return _resolve_path(raw)


def data_dir() -> Path:
    raw = os.environ.get(DATA_DIR_ENV)
    if raw is None or str(raw).strip() == "":
        return DEFAULT_DATA_DIR
    return _resolve_path(raw)


def project_path(raw: str | os.PathLike[str]) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def outputs_path(*parts: str) -> Path:
    return output_dir().joinpath(*parts)


def data_path(*parts: str) -> Path:
    return data_dir().joinpath(*parts)


def _set_cache_env(name: str, value: Path, *, force: bool) -> None:
    if force or os.environ.get(name, "").strip() == "":
        os.environ[name] = str(value)


def configure_model_data_dir() -> Path:
    """Configure model-related downloads and generated metadata under this project."""
    model_dir = model_data_dir()
    hf_home = model_dir / "huggingface"
    hf_hub_cache = hf_home / "hub"
    hf_datasets_cache = hf_home / "datasets"
    torch_home = model_dir / "torch"
    wandb_cache = model_dir / "wandb" / "cache"
    wandb_artifacts = model_dir / "wandb" / "artifacts"
    local_output_dir = output_dir()
    local_data_dir = data_dir()
    wandb_dir = local_output_dir / "wandb"

    for path in (
        model_dir,
        hf_home,
        hf_hub_cache,
        hf_datasets_cache,
        torch_home,
        wandb_cache,
        wandb_artifacts,
        local_output_dir,
        local_data_dir,
        wandb_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    _set_cache_env("HF_HOME", hf_home, force=True)
    _set_cache_env("HF_HUB_CACHE", hf_hub_cache, force=True)
    _set_cache_env("HUGGINGFACE_HUB_CACHE", hf_hub_cache, force=True)
    _set_cache_env("HF_DATASETS_CACHE", hf_datasets_cache, force=True)
    _set_cache_env("TRANSFORMERS_CACHE", hf_hub_cache, force=True)
    _set_cache_env("TORCH_HOME", torch_home, force=True)
    _set_cache_env(OUTPUT_DIR_ENV, local_output_dir, force=True)
    _set_cache_env(DATA_DIR_ENV, local_data_dir, force=True)
    _set_cache_env("WANDB_DIR", wandb_dir, force=True)
    _set_cache_env("WANDB_CACHE_DIR", wandb_cache, force=True)
    _set_cache_env("WANDB_ARTIFACT_DIR", wandb_artifacts, force=True)
    return model_dir


def hf_hub_cache_dir() -> Path:
    configure_model_data_dir()
    raw = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if raw is None or raw.strip() == "":
        return model_data_dir() / "huggingface" / "hub"
    return _resolve_path(raw)


def transformers_cache_dir() -> Path:
    return hf_hub_cache_dir()


def hf_datasets_cache_dir() -> Path:
    configure_model_data_dir()
    raw = os.environ.get("HF_DATASETS_CACHE")
    if raw is None or raw.strip() == "":
        return model_data_dir() / "huggingface" / "datasets"
    return _resolve_path(raw)


configure_model_data_dir()
