#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    get_cached_runtime,
)
from irodori_tts.model_cache import hf_hub_cache_dir, outputs_path, project_path

DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_MODEL_NAME = "Irodori-TTS-Reference-Audio"
DEFAULT_STYLE_NAME = "reference_audio"
DEFAULT_PORT = 6000
NO_REFERENCE_SPEAKER = "no_reference"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
EMOTION_NORMAL = "normal"
EMOTION_ANGRY = "angry"
EMOTION_SAD = "sad"
EMOTION_HAPPY = "happy"
EMOTION_CHOICES = (EMOTION_NORMAL, EMOTION_ANGRY, EMOTION_SAD, EMOTION_HAPPY)
EMOTION_ALIASES = {
    EMOTION_NORMAL: ("normal", "neutral", "default", "plain", "base", "普通", "通常", "平常"),
    EMOTION_ANGRY: ("angry", "anger", "mad", "rage", "ikari", "怒", "怒り", "激怒"),
    EMOTION_SAD: ("sad", "sorrow", "cry", "kanashi", "哀", "悲", "悲しみ", "泣"),
    EMOTION_HAPPY: (
        "happy",
        "joy",
        "fun",
        "smile",
        "laugh",
        "raku",
        "楽",
        "喜",
        "喜び",
        "嬉",
    ),
}
EMOTION_FACE_MARKERS = {
    EMOTION_ANGRY: (
        "😠",
        "😡",
        "🤬",
        "💢",
        "(#",
        "＃",
        "ಠ_ಠ",
        "｀Д´",
        "`Д´",
        "٩(๑`^´๑)۶",
    ),
    EMOTION_SAD: (
        "😢",
        "😭",
        "😿",
        "🥲",
        "T_T",
        "；；",
        ";_;",
        "orz",
        "(´；ω；`)",
        "(´・ω・｀)",
    ),
    EMOTION_HAPPY: (
        "😊",
        "😄",
        "😆",
        "😁",
        "😂",
        "🤣",
        "🙂",
        "^_^",
        "^^",
        "(*´",
        "ヽ(=´▽`=)ﾉ",
    ),
}
_ALIAS_TO_EMOTION = {
    alias.casefold(): emotion
    for emotion, aliases in EMOTION_ALIASES.items()
    for alias in aliases
}
_EMOTION_TAG_ALTERNATION = "|".join(
    re.escape(alias)
    for alias in sorted(_ALIAS_TO_EMOTION, key=len, reverse=True)
)
EMOTION_TAG_RE = re.compile(
    rf"(?:\[(?P<bracket>{_EMOTION_TAG_ALTERNATION})\]"
    rf"|【(?P<wide>{_EMOTION_TAG_ALTERNATION})】"
    rf"|（(?P<paren_wide>{_EMOTION_TAG_ALTERNATION})）"
    rf"|\((?P<paren>{_EMOTION_TAG_ALTERNATION})\)"
    rf"|<(?P<angle>{_EMOTION_TAG_ALTERNATION})>"
    rf"|:(?P<colon>{_EMOTION_TAG_ALTERNATION}):)",
    re.IGNORECASE,
)
STRIP_FACE_MARKERS = tuple(
    marker
    for markers in EMOTION_FACE_MARKERS.values()
    for marker in markers
    if marker not in {"(#", "＃", "(*´"}
)


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    path = Path(checkpoint).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return str(project_path(path))

    resolved = hf_hub_download(
        repo_id=checkpoint,
        filename="model.safetensors",
        cache_dir=str(hf_hub_cache_dir()),
    )
    print(f"[irodori-wrapper] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _speaker_name_for_path(path: Path) -> str:
    return path.stem.strip() or path.name


def _normalized_emotion_text(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.casefold())


def _emotion_from_alias(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalized_emotion_text(value)
    if normalized in _ALIAS_TO_EMOTION:
        return _ALIAS_TO_EMOTION[normalized]
    return None


def _emotion_for_reference_name(value: str) -> str | None:
    normalized = _normalized_emotion_text(value)
    for emotion, aliases in EMOTION_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalized_emotion_text(alias)
            if alias_norm and alias_norm in normalized:
                return emotion
    return None


def _collect_reference_audio(args: argparse.Namespace) -> dict[str, Path]:
    refs: dict[str, Path] = {}

    def add_path(path: Path) -> None:
        resolved = project_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"reference audio not found: {resolved}")
        key = _speaker_name_for_path(resolved)
        if key in refs:
            key = f"{key}_{len(refs) + 1}"
        refs[key] = resolved

    for item in args.reference_audio or []:
        add_path(Path(item))

    if args.reference_audio_dir:
        ref_dir = project_path(args.reference_audio_dir)
        if not ref_dir.is_dir():
            raise NotADirectoryError(f"reference audio dir not found: {ref_dir}")
        for path in sorted(ref_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
                add_path(path)

    return refs


def _build_emotion_reference_map(references: dict[str, Path]) -> dict[str, Path]:
    emotion_refs: dict[str, Path] = {}
    for name, path in references.items():
        emotion = _emotion_for_reference_name(name) or _emotion_for_reference_name(path.stem)
        if emotion is not None and emotion not in emotion_refs:
            emotion_refs[emotion] = path

    if references and EMOTION_NORMAL not in emotion_refs:
        emotion_refs[EMOTION_NORMAL] = next(iter(references.values()))
    return emotion_refs


def _detect_text_emotion(text: str) -> str | None:
    match = EMOTION_TAG_RE.search(text)
    if match is not None:
        for value in match.groupdict().values():
            emotion = _emotion_from_alias(value)
            if emotion is not None:
                return emotion

    for emotion in (EMOTION_ANGRY, EMOTION_SAD, EMOTION_HAPPY):
        if any(marker in text for marker in EMOTION_FACE_MARKERS[emotion]):
            return emotion
    return None


def _strip_emotion_markers(text: str) -> str:
    cleaned = EMOTION_TAG_RE.sub("", text)
    for marker in STRIP_FACE_MARKERS:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([。、，,.!！?？])", r"\1", cleaned)
    return cleaned.strip() or text


def _audio_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to encode wrapper responses as wav.") from exc

    wav = audio.detach().cpu().float()
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    if wav.ndim != 1:
        raise RuntimeError(f"unexpected generated audio shape: {tuple(wav.shape)}")

    buffer = BytesIO()
    sf.write(buffer, wav.numpy(), int(sample_rate), format="WAV")
    return buffer.getvalue()


def _build_runtime_key(args: argparse.Namespace) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(args.model_device),
        codec_repo=str(args.codec_repo),
        model_precision=str(args.model_precision),
        codec_device=str(args.codec_device),
        codec_precision=str(args.codec_precision),
        codec_deterministic_encode=bool(args.codec_deterministic_encode),
        codec_deterministic_decode=bool(args.codec_deterministic_decode),
        enable_watermark=bool(args.enable_watermark),
        compile_model=bool(args.compile_model),
        compile_dynamic=bool(args.compile_dynamic),
    )


def create_app(args: argparse.Namespace) -> FastAPI:
    references = _collect_reference_audio(args)
    emotion_references = _build_emotion_reference_map(references)
    runtime_key = _build_runtime_key(args)
    app = FastAPI(title="Irodori-TTS SBV2-compatible wrapper")
    shutdown_requested = threading.Event()
    job_cancel_requested = threading.Event()
    job_lock = threading.Lock()
    infer_lock = threading.Lock()
    state: dict[str, int] = {"generation": 0, "active": 0}
    server_holder: dict[str, Any] = {"server": None}

    def raise_validation_error(msg: str, param: str) -> None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"type": "invalid_params", "msg": msg, "loc": ["query", param]}],
        )

    def current_generation() -> int:
        with job_lock:
            return int(state["generation"])

    def current_models_info() -> dict[str, dict[str, Any]]:
        speakers = {name: index for index, name in enumerate(references)}
        if not speakers:
            speakers[NO_REFERENCE_SPEAKER] = 0
        styles = {str(args.style_name): 0}
        for emotion in EMOTION_CHOICES:
            if emotion in emotion_references and emotion not in styles:
                styles[emotion] = len(styles)
        model_path = Path("model_assets") / str(args.model_name) / "model.safetensors"
        config_path = Path("model_assets") / str(args.model_name) / "config.json"
        return {
            "0": {
                "config_path": str(config_path),
                "model_path": str(model_path),
                "device": str(args.model_device),
                "spk2id": speakers,
                "id2spk": {str(value): key for key, value in speakers.items()},
                "style2id": styles,
            }
        }

    def resolve_reference(
        speaker_name: str | None,
        speaker_id: int,
        reference_audio_path: str | None,
        *,
        detected_emotion: str | None,
        requested_style: str | None,
    ) -> Path | None:
        if reference_audio_path:
            ref_path = project_path(reference_audio_path)
            if not ref_path.is_file():
                raise_validation_error(f"reference_audio_path={ref_path} not found", "reference_audio_path")
            return ref_path
        if (
            args.emotion_auto
            and args.emotion_overrides_speaker
            and detected_emotion is not None
            and detected_emotion in emotion_references
        ):
            return emotion_references[detected_emotion]
        if not references:
            if args.allow_no_ref:
                return None
            raise_validation_error(
                "reference audio is required. Start wrapper with --reference-audio or --reference-audio-dir.",
                "speaker_name",
            )
        if speaker_name and speaker_name in references:
            return references[speaker_name]
        if speaker_name and speaker_name not in {"", NO_REFERENCE_SPEAKER}:
            raise_validation_error(f"speaker_name={speaker_name} not found", "speaker_name")
        style_emotion = _emotion_from_alias(requested_style)
        if style_emotion is not None and style_emotion in emotion_references:
            return emotion_references[style_emotion]
        refs_by_index = list(references.values())
        if (
            not args.emotion_overrides_speaker
            and 0 <= int(speaker_id) < len(refs_by_index)
        ):
            return refs_by_index[int(speaker_id)]
        if (
            args.emotion_auto
            and detected_emotion is not None
            and detected_emotion in emotion_references
        ):
            return emotion_references[detected_emotion]
        if 0 <= int(speaker_id) < len(refs_by_index):
            return refs_by_index[int(speaker_id)]
        if args.default_emotion in emotion_references:
            return emotion_references[args.default_emotion]
        return next(iter(references.values()))

    @app.get("/models/info")
    def get_loaded_models_info() -> dict[str, dict[str, Any]]:
        return current_models_info()

    @app.post("/models/refresh")
    def refresh_models() -> dict[str, dict[str, Any]]:
        return current_models_info()

    @app.api_route("/voice", methods=["GET", "POST"])
    def voice(
        request: Request,
        text: str = Query(..., min_length=1),
        model_name: str | None = Query(None),
        speaker_name: str | None = Query(None),
        speaker_id: int = Query(0),
        style: str | None = Query(DEFAULT_STYLE_NAME),
        style_weight: float = Query(1.0),
        length: float = Query(1.0),
        noise: float = Query(0.6),
        noisew: float = Query(0.8),
        sdp_ratio: float = Query(0.2),
        language: str = Query("JP"),
        reference_audio_path: str | None = Query(None),
    ) -> Response:
        del request, style_weight, length, noise, noisew, sdp_ratio, language
        if shutdown_requested.is_set():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="server is shutting down",
            )
        if job_cancel_requested.is_set():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="job is cancelled",
            )
        if model_name and model_name != args.model_name:
            raise_validation_error(f"model_name={model_name} not found", "model_name")
        if style and style != args.style_name and _emotion_from_alias(style) is None:
            raise_validation_error(f"style={style} not found", "style")

        detected_emotion = _detect_text_emotion(text) if args.emotion_auto else None
        synthesis_text = _strip_emotion_markers(text) if args.emotion_strip_markers else text

        ref_wav = resolve_reference(
            speaker_name,
            speaker_id,
            reference_audio_path,
            detected_emotion=detected_emotion,
            requested_style=style,
        )

        request_generation = current_generation()
        with job_lock:
            state["active"] += 1
        try:
            with infer_lock:
                if job_cancel_requested.is_set() or current_generation() != request_generation:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="job was cancelled",
                    )
                runtime, reloaded = get_cached_runtime(runtime_key)
                print(
                    "[irodori-wrapper] synthesize: runtime={} text={!r} ref={}".format(
                        "reloaded" if reloaded else "reused",
                        synthesis_text,
                        ref_wav,
                    ),
                    flush=True,
                )
                if detected_emotion is not None:
                    print(
                        f"[irodori-wrapper] detected emotion={detected_emotion}",
                        flush=True,
                    )
                result = runtime.synthesize(
                    SamplingRequest(
                        text=synthesis_text,
                        ref_wav=None if ref_wav is None else str(ref_wav),
                        ref_latent=None,
                        no_ref=ref_wav is None,
                        ref_normalize_db=args.ref_normalize_db,
                        ref_ensure_max=bool(args.ref_ensure_max),
                        num_candidates=1,
                        decode_mode="sequential",
                        seconds=float(args.seconds),
                        max_ref_seconds=args.max_ref_seconds,
                        max_text_len=args.max_text_len,
                        num_steps=int(args.num_steps),
                        seed=None,
                        cfg_guidance_mode=str(args.cfg_guidance_mode),
                        cfg_scale_text=float(args.cfg_scale_text),
                        cfg_scale_speaker=float(args.cfg_scale_speaker),
                        cfg_scale_caption=0.0,
                        cfg_scale=None,
                        cfg_min_t=float(args.cfg_min_t),
                        cfg_max_t=float(args.cfg_max_t),
                        context_kv_cache=bool(args.context_kv_cache),
                        trim_tail=True,
                    ),
                    log_fn=lambda message: print(f"[irodori-wrapper] {message}", flush=True),
                )
        finally:
            with job_lock:
                state["active"] = max(0, state["active"] - 1)

        if job_cancel_requested.is_set() or current_generation() != request_generation:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="job was cancelled",
            )
        return Response(
            content=_audio_to_wav_bytes(result.audio, result.sample_rate),
            media_type="audio/wav",
        )

    @app.get("/tools/get_audio")
    def get_audio(path: str = Query(..., description="local wav path")) -> FileResponse:
        wav_path = project_path(path)
        if not wav_path.is_file():
            raise_validation_error(f"path={wav_path} not found", "path")
        if wav_path.suffix.lower() != ".wav":
            raise_validation_error(f"wav file not found in {wav_path}", "path")
        return FileResponse(str(wav_path), media_type="audio/wav")

    @app.get("/status")
    def get_status() -> dict[str, Any]:
        return {
            "devices": ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
            "model_device": args.model_device,
            "codec_device": args.codec_device,
            "reference_count": len(references),
            "emotion_auto": bool(args.emotion_auto),
            "emotion_references": {
                emotion: str(path) for emotion, path in emotion_references.items()
            },
            "checkpoint": runtime_key.checkpoint,
        }

    @app.post("/jobs/cancel")
    def cancel_jobs() -> dict[str, Any]:
        with job_lock:
            state["generation"] += 1
            active = state["active"]
        job_cancel_requested.set()
        print(f"[irodori-wrapper] job cancel requested: active={active}", flush=True)
        return {"ok": True, "active_voice_requests": active}

    @app.post("/jobs/resume")
    def resume_jobs() -> dict[str, bool]:
        job_cancel_requested.clear()
        print("[irodori-wrapper] accepting requests", flush=True)
        return {"ok": True}

    @app.get("/jobs/status")
    def jobs_status() -> dict[str, Any]:
        with job_lock:
            return {
                "cancel_requested": job_cancel_requested.is_set(),
                "generation": state["generation"],
                "active_voice_requests": state["active"],
            }

    @app.post("/shutdown")
    def shutdown() -> dict[str, bool]:
        shutdown_requested.set()
        print("[irodori-wrapper] graceful shutdown requested", flush=True)

        def request_stop() -> None:
            server = server_holder.get("server")
            if server is not None:
                server.should_exit = True

        timer = threading.Timer(0.2, request_stop)
        timer.daemon = True
        timer.start()
        return {"ok": True}

    app.state.irodori_server_holder = server_holder
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SBV2-compatible HTTP wrapper for Irodori-TTS reference-audio inference."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument("--reference-audio", action="append", default=[])
    parser.add_argument("--reference-audio-dir", default=None)
    parser.add_argument("--allow-no-ref", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--style-name", default=DEFAULT_STYLE_NAME)
    parser.add_argument("--default-emotion", choices=EMOTION_CHOICES, default=EMOTION_NORMAL)
    parser.add_argument("--emotion-auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emotion-strip-markers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--emotion-overrides-speaker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let text emotion markers choose reference audio even when speaker_name is set.",
    )
    parser.add_argument("--model-device", default=default_runtime_device())
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-device", default=default_runtime_device())
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-deterministic-encode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--codec-deterministic-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-watermark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--max-ref-seconds", type=float, default=30.0)
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-guidance-mode", choices=["independent", "joint", "alternating"], default="independent")
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument("--context-kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ref-normalize-db", type=float, default=-16.0)
    parser.add_argument("--ref-ensure-max", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    outputs_path("wrapper").mkdir(parents=True, exist_ok=True)
    app = create_app(args)
    print(f"[irodori-wrapper] listen: http://127.0.0.1:{args.port}", flush=True)
    print(f"[irodori-wrapper] docs: http://127.0.0.1:{args.port}/docs", flush=True)
    server = uvicorn.Server(
        uvicorn.Config(app, host=str(args.host), port=int(args.port), log_level="warning")
    )
    app.state.irodori_server_holder["server"] = server
    server.run()
    time.sleep(0.1)


if __name__ == "__main__":
    main()
