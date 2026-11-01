"""Irodori reference speech -> Seed-VC v1 singing conversion, as sequential subprocesses.

Each model runs in its own venv and its own process, so Irodori and Seed-VC are
never resident on the GPU together and their Torch/Transformers pins never
meet. This module is standard-library only and can be driven from the CLI or
the Gradio UI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGING_DIR = PROJECT_ROOT / ".singing"
SEEDVC_DIR = SINGING_DIR / "seed-vc"
SEEDVC_PYTHON = SINGING_DIR / "venv" / "Scripts" / "python.exe"
IRODORI_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
ASSET_MANIFEST = SINGING_DIR / "seedvc_assets.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "singing"
DEFAULT_HF_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"

LogFn = Callable[[str], None]


class PipelineError(RuntimeError):
    """A step failed; ``log_path`` holds its full output."""

    def __init__(self, message: str, *, step: str = "", log_path: Path | None = None):
        super().__init__(message)
        self.step = step
        self.log_path = log_path


@dataclass
class SingingOptions:
    source_wav: str
    reference_wav: str | None = None  # existing target reference; excludes TTS options
    tts_text: str | None = None
    tts_caption: str | None = None
    tts_checkpoint: str | None = None  # local .safetensors/.pt; overrides tts_hf_checkpoint
    tts_hf_checkpoint: str = DEFAULT_HF_CHECKPOINT
    tts_lora_adapter: str | None = None
    tts_speaker_wav: str | None = None  # Irodori speaker prompt; omitted means --no-ref
    tts_seed: int | None = 0
    tts_num_steps: int | None = None
    tts_seconds: float | None = None
    tts_extra_args: list[str] = field(default_factory=list)
    source_start: float = 0.0
    source_duration: float | None = None
    diffusion_steps: int = 30
    length_adjust: float = 1.0
    inference_cfg_rate: float = 0.7
    f0_condition: bool = True
    auto_f0_adjust: bool = False
    semitone_shift: int = 0
    fp16: bool = True
    seedvc_checkpoint: str | None = None  # defaults to the pinned v1 singing checkpoint
    seedvc_config: str | None = None
    output_dir: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str | None = None
    offline: bool = True
    f0_report: bool = True


def _is_set(value: str | None) -> bool:
    return value is not None and str(value).strip() != ""


def validate(options: SingingOptions) -> list[str]:
    """Return human-readable problems; empty means the request is runnable."""
    problems = []
    if not _is_set(options.source_wav):
        problems.append("source vocal WAV is required")
    elif not Path(options.source_wav).is_file():
        problems.append(f"source vocal not found: {options.source_wav}")
    if _is_set(options.reference_wav) == _is_set(options.tts_text):
        problems.append("give exactly one of reference_wav (existing WAV) or tts_text (Irodori)")
    if _is_set(options.reference_wav) and not Path(options.reference_wav).is_file():
        problems.append(f"reference WAV not found: {options.reference_wav}")
    for label, value in (("TTS checkpoint", options.tts_checkpoint),
                         ("TTS speaker WAV", options.tts_speaker_wav),
                         ("Seed-VC checkpoint", options.seedvc_checkpoint),
                         ("Seed-VC config", options.seedvc_config)):
        if _is_set(value) and not Path(value).is_file():
            problems.append(f"{label} not found: {value}")
    if _is_set(options.tts_lora_adapter) and not Path(options.tts_lora_adapter).is_dir():
        problems.append(f"LoRA adapter directory not found: {options.tts_lora_adapter}")
    if not 0.0 < options.length_adjust <= 4.0:
        problems.append("length_adjust must be in (0, 4]")
    if options.diffusion_steps < 1:
        problems.append("diffusion_steps must be >= 1")
    if options.source_start < 0 or (options.source_duration is not None and options.source_duration <= 0):
        problems.append("source_start must be >= 0 and source_duration > 0")
    if options.f0_condition is False and not _is_set(options.seedvc_config):
        problems.append("f0_condition=false needs an explicit non-f0 seedvc_config; "
                        "the singing model is never swapped for a speech model automatically")
    return problems


def check_environment(need_tts: bool) -> list[str]:
    problems = []
    if not SEEDVC_PYTHON.is_file():
        problems.append(f"Seed-VC venv missing: {SEEDVC_PYTHON} (run setup_singing.bat)")
    if not (SEEDVC_DIR / "inference.py").is_file():
        problems.append(f"Seed-VC source missing: {SEEDVC_DIR} (run setup_singing.bat)")
    if not ASSET_MANIFEST.is_file():
        problems.append(f"Seed-VC assets not prepared: {ASSET_MANIFEST} (run setup_singing.bat)")
    if need_tts and not IRODORI_PYTHON.is_file():
        problems.append(f"Irodori venv missing: {IRODORI_PYTHON} (run setup_singing.bat)")
    return problems


def default_seedvc_model() -> tuple[str, str]:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    return manifest["singing_checkpoint"], manifest["singing_config"]


def irodori_command(options: SingingOptions, output_wav: Path) -> list[str]:
    command = [str(IRODORI_PYTHON), str(PROJECT_ROOT / "infer.py")]
    if _is_set(options.tts_checkpoint):
        command += ["--checkpoint", str(Path(options.tts_checkpoint).resolve())]
    else:
        command += ["--hf-checkpoint", options.tts_hf_checkpoint]
    command += ["--text", str(options.tts_text)]
    if _is_set(options.tts_caption):
        command += ["--caption", str(options.tts_caption)]
    if _is_set(options.tts_lora_adapter):
        command += ["--lora-adapter", str(Path(options.tts_lora_adapter).resolve())]
    if _is_set(options.tts_speaker_wav):
        command += ["--ref-wav", str(Path(options.tts_speaker_wav).resolve())]
    else:
        command += ["--no-ref"]
    if options.tts_seed is not None:
        command += ["--seed", str(int(options.tts_seed))]
    if options.tts_num_steps is not None:
        command += ["--num-steps", str(int(options.tts_num_steps))]
    if options.tts_seconds is not None:
        command += ["--seconds", str(float(options.tts_seconds))]
    command += list(options.tts_extra_args)
    command += ["--output-wav", str(output_wav)]
    return command


def seedvc_command(options: SingingOptions, run_dir: Path, checkpoint: str, config: str) -> list[str]:
    return [
        str(SEEDVC_PYTHON), str(PROJECT_ROOT / "singing_vc" / "seedvc_runner.py"),
        "--seedvc-dir", str(SEEDVC_DIR),
        "--source", str(run_dir / "source.wav"),
        "--target", str(run_dir / "reference.wav"),
        "--output-wav", str(run_dir / "converted.wav"),
        "--report-json", str(run_dir / "seedvc_report.json"),
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--config", str(Path(config).resolve()),
        "--diffusion-steps", str(int(options.diffusion_steps)),
        "--length-adjust", repr(float(options.length_adjust)),
        "--inference-cfg-rate", repr(float(options.inference_cfg_rate)),
        "--f0-condition", str(bool(options.f0_condition)),
        "--auto-f0-adjust", str(bool(options.auto_f0_adjust)),
        "--semi-tone-shift", str(int(options.semitone_shift)),
        "--fp16", str(bool(options.fp16)),
        "--offline", str(bool(options.offline)),
    ]


def run_step(name: str, command: list[str], log_dir: Path, *, cwd: Path = PROJECT_ROOT,
             env: dict[str, str] | None = None, log: LogFn = print) -> float:
    """Run one subprocess, tee its output to ``log_dir/<name>.log``; raise PipelineError on failure."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log(f"[{name}] start")
    started = time.perf_counter()
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(command) + "\n")
        try:
            process = subprocess.Popen(
                command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
            )
        except OSError as exc:
            raise PipelineError(f"{name}: could not start {command[0]}: {exc}", step=name, log_path=log_path) from exc
        with process:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                line = line.rstrip()
                tail = (tail + [line])[-15:]
                log(f"[{name}] {line}")
            code = process.wait()
    elapsed = time.perf_counter() - started
    if code != 0:
        raise PipelineError(
            f"{name} failed with exit code {code} (log: {log_path})\n" + "\n".join(tail),
            step=name, log_path=log_path,
        )
    log(f"[{name}] done in {elapsed:.1f}s")
    return elapsed


def child_environment(offline: bool) -> dict[str, str]:
    sys.path.insert(0, str(PROJECT_ROOT / "irodori_tts"))
    try:
        from local_env import configure_project_environment  # keeps caches/temp in this checkout
    finally:
        sys.path.pop(0)
    configure_project_environment()
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10, check=False)
    except OSError:
        return None
    return result.stdout.strip() or None


def make_run_dir(options: SingingOptions) -> Path:
    label = re.sub(r"[^0-9A-Za-z_-]+", "_", options.run_name or "").strip("_")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(options.output_dir).expanduser().resolve()
    run_dir = root / (f"{stamp}_{label}" if label else stamp)
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{run_dir.name}-{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def run_pipeline(options: SingingOptions, log: LogFn = print) -> dict:
    problems = validate(options) + check_environment(need_tts=_is_set(options.tts_text))
    if problems:
        raise PipelineError("Invalid request:\n- " + "\n- ".join(problems), step="validate")
    checkpoint, config = default_seedvc_model()
    checkpoint = options.seedvc_checkpoint or checkpoint
    config = options.seedvc_config or config

    run_dir = make_run_dir(options)
    logs = run_dir / "logs"
    env = child_environment(options.offline)
    metadata: dict = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "options": asdict(options),
        "reference_mode": "irodori_tts" if _is_set(options.tts_text) else "existing_wav",
        "revisions": {"irodori_checkout": git_revision(PROJECT_ROOT), "seed_vc": git_revision(SEEDVC_DIR)},
        "seedvc_model": {"checkpoint": checkpoint, "config": config},
        "commands": {},
        "step_seconds": {},
        "validation_scope": "technical checks only; voice likeness and singing quality need human listening",
    }
    metadata_path = run_dir / "metadata.json"

    def save() -> None:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def step(name: str, command: list[str], **kwargs) -> None:
        metadata["commands"][name] = command
        save()
        metadata["step_seconds"][name] = round(run_step(name, command, logs, env=env, log=log, **kwargs), 2)

    tool = str(PROJECT_ROOT / "singing_vc" / "audio_tool.py")
    try:
        prepare = [str(SEEDVC_PYTHON), tool, "prepare", "--input", str(Path(options.source_wav).resolve()),
                   "--output", str(run_dir / "source.wav"), "--start", repr(float(options.source_start)),
                   "--json", str(run_dir / "source_prepare.json")]
        if options.source_duration is not None:
            prepare += ["--duration", repr(float(options.source_duration))]
        step("prepare_source", prepare)

        if _is_set(options.tts_text):
            # Irodori process exits before Seed-VC starts, releasing its GPU memory.
            step("irodori_reference", irodori_command(options, run_dir / "reference.wav"))
        else:
            shutil.copyfile(Path(options.reference_wav).resolve(), run_dir / "reference.wav")
            metadata["reference_source"] = str(Path(options.reference_wav).resolve())
        if not (run_dir / "reference.wav").is_file():
            raise PipelineError("reference.wav was not produced", step="irodori_reference")

        step("seedvc_singing", seedvc_command(options, run_dir, checkpoint, config), cwd=SEEDVC_DIR)
        metadata["seedvc_report"] = json.loads((run_dir / "seedvc_report.json").read_text(encoding="utf-8"))

        analyze = [str(SEEDVC_PYTHON), tool, "analyze", "--source", str(run_dir / "source.wav"),
                   "--reference", str(run_dir / "reference.wav"), "--converted", str(run_dir / "converted.wav"),
                   "--semitone-shift", str(int(options.semitone_shift)),
                   "--length-adjust", repr(float(options.length_adjust)), "--json", str(run_dir / "analysis.json")]
        if options.auto_f0_adjust:
            analyze.append("--auto-f0-adjust")
        if not options.f0_report:
            analyze.append("--skip-f0")
        step("analyze", analyze)
        analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
        metadata["analysis"] = analysis
        converted = analysis["converted"]
        checks = {
            "sample_rate_44100": converted["sample_rate"] == 44100,
            "finite": converted["finite"],
            "non_silent": converted["non_silent"],
            "duration_matches_source_within_5pct":
                abs(analysis["duration_ratio_converted_to_source"] - options.length_adjust) <= 0.05 * options.length_adjust,
        }
        metadata["checks"] = checks
        metadata["status"] = "succeeded" if all(checks.values()) else "completed_with_failed_checks"
    except Exception as exc:  # record any failure, then let the caller report it
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        failed_step=getattr(exc, "step", ""))
        raise
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        save()
        write_summary_ja(run_dir, metadata)
    log(f"[done] {metadata['status']}: {run_dir / 'converted.wav'}")
    return metadata


def write_summary_ja(run_dir: Path, metadata: dict) -> None:
    analysis = metadata.get("analysis", {})
    converted = analysis.get("converted", {})
    f0 = analysis.get("f0_source_vs_converted", {})
    report = metadata.get("seedvc_report", {})
    opts = metadata["options"]
    lines = [
        f"# 歌唱変換 実行結果 ({metadata['status']})",
        "",
        f"- 実行フォルダ: `{run_dir}`",
        f"- 参照音声: {'Irodori-TTS で生成' if metadata['reference_mode'] == 'irodori_tts' else '既存 WAV'}",
        f"- Seed-VC 設定: f0-condition={opts['f0_condition']}, auto-f0-adjust={opts['auto_f0_adjust']}, "
        f"semitone={opts['semitone_shift']}, length-adjust={opts['length_adjust']}, "
        f"diffusion-steps={opts['diffusion_steps']}, cfg-rate={opts['inference_cfg_rate']}, fp16={opts['fp16']}",
        f"- 実行デバイス: {report.get('device', '未取得')} / 変換時間 {report.get('elapsed_seconds', '-')} 秒",
    ]
    if converted:
        lines += [
            f"- 出力: {converted['sample_rate']} Hz, {converted['duration_seconds']} 秒, "
            f"peak {converted['peak']}, RMS {converted['rms_dbfs']} dBFS, finite={converted['finite']}, "
            f"non-silent={converted['non_silent']}",
            f"- 長さ比 (出力/入力): {analysis.get('duration_ratio_converted_to_source')}",
        ]
    if "median_abs_cents_error" in f0:
        lines.append(
            f"- F0 比較 (pYIN): 中央値誤差 {f0['median_abs_cents_error']} cent, ±50cent 以内 "
            f"{f0['within_50_cents_ratio']:.1%}, オクターブ誤り {f0['octave_error_ratio']:.1%}, "
            f"相関 {f0['f0_correlation']}, 有声区間 IoU {f0['voicing_overlap_iou']}"
        )
    for key in ("not_comparable_reason", "cents_not_comparable_reason"):
        if key in f0:
            lines.append(f"- F0 比較の制限: {f0[key]}")
    if f0:
        lines.append("- F0 指標は同じフレーム位置での音高一致のみを示します。発音・立ち上がりのタイミングやずれは別途確認が必要です。")
    if metadata.get("error"):
        lines += ["", "## エラー", "```", metadata["error"], "```"]
    lines += ["", "注意: 上記は技術的な客観指標のみです。声の似ている度合い・歌唱品質は試聴による確認が必要です。"]
    (run_dir / "summary_ja.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
