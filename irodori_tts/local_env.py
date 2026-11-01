"""Keep this checkout's downloads and runtime caches inside the workspace."""

from __future__ import annotations

import json
import os
import ssl
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA = PROJECT_ROOT / "model_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CERT_BUNDLE = MODEL_DATA / "certs" / "windows-ca.pem"
FFMPEG_BIN = PROJECT_ROOT / ".tools" / "ffmpeg" / "bin"


def project_paths() -> dict[str, Path]:
    cache = PROJECT_ROOT / ".cache"
    hf_home = MODEL_DATA / "huggingface"
    return {
        "UV_PROJECT_ENVIRONMENT": PROJECT_ROOT / ".venv",
        "UV_CACHE_DIR": cache / "uv",
        "UV_PYTHON_INSTALL_DIR": MODEL_DATA / "python",
        "PIP_CACHE_DIR": cache / "pip",
        "XDG_CACHE_HOME": cache,
        "IRODORI_TTS_MODEL_DIR": MODEL_DATA,
        "IRODORI_TTS_OUTPUT_DIR": OUTPUT_DIR,
        "IRODORI_TTS_DATA_DIR": PROJECT_ROOT / "data",
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "HF_DATASETS_CACHE": hf_home / "datasets",
        "HF_XET_CACHE": hf_home / "xet",
        "TORCH_HOME": MODEL_DATA / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "MPLCONFIGDIR": cache / "matplotlib",
        "NUMBA_CACHE_DIR": cache / "numba",
        "WANDB_DIR": OUTPUT_DIR / "wandb",
        "WANDB_CACHE_DIR": MODEL_DATA / "wandb" / "cache",
        "WANDB_ARTIFACT_DIR": MODEL_DATA / "wandb" / "artifacts",
        "WANDB_CONFIG_DIR": MODEL_DATA / "wandb" / "config",
        "GRADIO_TEMP_DIR": OUTPUT_DIR / "gradio_tmp",
        "TEMP": cache / "tmp",
        "TMP": cache / "tmp",
        "TMPDIR": cache / "tmp",
    }


def _export_windows_ca_bundle() -> None:
    if os.name != "nt" or CERT_BUNDLE.exists():
        return
    blocks: set[str] = set()
    for store in ("ROOT", "CA"):
        for raw, encoding, _trust in ssl.enum_certificates(store):
            if encoding == "x509_asn":
                blocks.add(ssl.DER_cert_to_PEM_cert(raw))
    if blocks:
        CERT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=CERT_BUNDLE.parent, delete=False
        ) as stream:
            stream.write("".join(sorted(blocks)))
            temporary = Path(stream.name)
        temporary.replace(CERT_BUNDLE)


def configure_project_environment() -> dict[str, str]:
    """Set process-local paths before importing HF, Torch, datasets or Gradio."""
    paths = project_paths()
    for name, path in paths.items():
        # uv owns the virtualenv itself; do not create an empty .venv here.
        if name != "UV_PROJECT_ENVIRONMENT":
            path.mkdir(parents=True, exist_ok=True)
    values = {name: str(path) for name, path in paths.items()}
    values.update(UV_SYSTEM_CERTS="true", UV_LINK_MODE="copy", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    if (FFMPEG_BIN / "ffmpeg.exe").is_file():
        other_paths = [
            entry for entry in os.environ.get("PATH", "").split(os.pathsep)
            if entry.casefold() != str(FFMPEG_BIN).casefold()
        ]
        values["PATH"] = os.pathsep.join([str(FFMPEG_BIN), *other_paths])
    os.environ.update(values)
    # Transformers 5 uses HF_HOME; discard an inherited obsolete cache override.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.pop("UV_NATIVE_TLS", None)
    tempfile.tempdir = str(paths["TEMP"])
    _export_windows_ca_bundle()
    if CERT_BUNDLE.exists():
        for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "GIT_SSL_CAINFO"):
            values[name] = str(CERT_BUNDLE)
        os.environ.update(values)
    return values


if __name__ == "__main__":
    print(json.dumps(configure_project_environment(), ensure_ascii=True))
