"""Audio preparation and objective checks, run inside the Seed-VC venv.

``prepare`` copies (optionally trims) an input into the run folder as WAV so
upstream code never sees awkward file names. ``analyze`` writes waveform facts
and an F0 comparison (librosa pYIN) between source and converted audio. These
numbers prove technical behaviour only; they say nothing about perceived
voice likeness or singing quality.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SILENCE_DBFS = -60.0
F0_SR = 16000
F0_HOP = 160  # 10 ms frames


def prepare(source: Path, output: Path, start: float, duration: float | None) -> dict:
    info = sf.info(str(source))
    first = int(round(max(start, 0.0) * info.samplerate))
    if first >= info.frames:
        raise ValueError(f"start {start}s is beyond the end of {source} ({info.frames / info.samplerate:.2f}s)")
    frames = -1 if duration is None else int(round(duration * info.samplerate))
    audio, sample_rate = sf.read(str(source), start=first, frames=frames, always_2d=True, dtype="float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio, sample_rate, subtype="PCM_16" if info.subtype == "PCM_16" else "FLOAT")
    return {"input": str(source), "output": str(output), "start_seconds": first / sample_rate,
            "duration_seconds": audio.shape[0] / sample_rate, "input_duration_seconds": info.frames / info.samplerate}


def waveform_facts(path: Path) -> dict:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    finite = bool(np.isfinite(mono).all())
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size and finite else 0.0
    rms_dbfs = 20 * math.log10(rms) if rms > 0 else -math.inf
    return {
        "path": str(path), "sample_rate": int(sample_rate), "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]), "duration_seconds": round(audio.shape[0] / sample_rate, 4),
        "finite": finite, "peak": round(float(np.abs(audio).max()) if audio.size else 0.0, 5) if finite else None,
        "clipped_sample_ratio": round(float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0, 6),
        "rms_dbfs": round(rms_dbfs, 2) if math.isfinite(rms_dbfs) else None,
        "non_silent": bool(finite and rms_dbfs > SILENCE_DBFS),
    }


def f0_contour(path: Path) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=F0_SR, mono=True)
    f0, voiced, _ = librosa.pyin(audio, fmin=65.0, fmax=1100.0, sr=F0_SR,
                                 frame_length=1024, hop_length=F0_HOP)
    return np.where(voiced, f0, np.nan)


def _round_or_none(value: float, digits: int) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


def compare_contours(src: np.ndarray, out: np.ndarray, semitone_shift: float,
                     length_adjust: float = 1.0, auto_f0_adjust: bool = False) -> dict:
    """Compare two F0 contours (NaN = unvoiced) frame by frame at the same index.

    Same-index frames only correspond when the output keeps the source timing,
    i.e. length_adjust == 1; otherwise no metrics are computed. Even then this
    measures pitch agreement under that alignment only, not phoneme/onset timing.
    With auto_f0_adjust upstream moves the pitch to the reference register, so
    the fixed requested shift is not a valid expectation and cents errors are skipped.
    """
    result = {
        "method": f"librosa.pyin {F0_SR} Hz, hop {F0_HOP}, 65-1100 Hz; frames aligned by index (no time warping)",
        "expected_shift_semitones": None if auto_f0_adjust else semitone_shift,
    }
    if not math.isclose(length_adjust, 1.0):
        result["not_comparable_reason"] = (
            f"length_adjust={length_adjust} changes output timing; same-index frames do not correspond")
        return result
    frames = min(len(src), len(out))
    src, out = src[:frames], out[:frames]
    both = ~np.isnan(src) & ~np.isnan(out)
    either = ~np.isnan(src) | ~np.isnan(out)
    result.update({
        "frames_compared": int(frames),
        "source_voiced_ratio": round(float(np.mean(~np.isnan(src))), 4) if frames else None,
        "converted_voiced_ratio": round(float(np.mean(~np.isnan(out))), 4) if frames else None,
        "voicing_overlap_iou": round(float(both.sum() / either.sum()), 4) if either.any() else None,
    })
    if both.sum() < 10:
        result["not_comparable_reason"] = "fewer than 10 frames voiced in both"
        return result
    log_src, log_out = np.log2(src[both]), np.log2(out[both])
    constant = np.ptp(log_src) == 0 or np.ptp(log_out) == 0
    result.update({
        "voiced_frames_both": int(both.sum()),
        # Pearson correlation is undefined for a flat contour; report null rather than NaN.
        "f0_correlation": None if constant else _round_or_none(np.corrcoef(log_src, log_out)[0, 1], 4),
        "source_median_hz": round(float(np.median(src[~np.isnan(src)])), 1),
        "converted_median_hz": round(float(np.median(out[~np.isnan(out)])), 1),
    })
    if auto_f0_adjust:
        result["cents_not_comparable_reason"] = (
            "auto_f0_adjust shifts pitch toward the reference; the requested semitone shift is not the expectation")
        return result
    cents = 1200 * (log_out - log_src) - 100 * semitone_shift
    # Octave errors are reported separately so they do not hide in the median.
    result.update({
        "median_abs_cents_error": round(float(np.median(np.abs(cents))), 1),
        "within_50_cents_ratio": round(float(np.mean(np.abs(cents) <= 50)), 4),
        "octave_error_ratio": round(float(np.mean(np.abs(cents) >= 600)), 4),
    })
    return result


def compare_f0(source: Path, converted: Path, semitone_shift: float,
               length_adjust: float = 1.0, auto_f0_adjust: bool = False) -> dict:
    if not math.isclose(length_adjust, 1.0):  # skip the slow pYIN pass when nothing can be compared
        return compare_contours(np.array([]), np.array([]), semitone_shift, length_adjust, auto_f0_adjust)
    return compare_contours(f0_contour(source), f0_contour(converted), semitone_shift,
                            length_adjust, auto_f0_adjust)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--start", type=float, default=0.0)
    prep.add_argument("--duration", type=float, default=None)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--source", type=Path, required=True)
    analyze.add_argument("--reference", type=Path, required=True)
    analyze.add_argument("--converted", type=Path, required=True)
    analyze.add_argument("--semitone-shift", type=float, default=0.0)
    analyze.add_argument("--length-adjust", type=float, default=1.0)
    analyze.add_argument("--auto-f0-adjust", action="store_true")
    analyze.add_argument("--skip-f0", action="store_true")
    for command in (prep, analyze):
        command.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        report = prepare(args.input, args.output, args.start, args.duration)
    else:
        report = {name: waveform_facts(getattr(args, name)) for name in ("source", "reference", "converted")}
        report["duration_ratio_converted_to_source"] = round(
            report["converted"]["duration_seconds"] / max(report["source"]["duration_seconds"], 1e-9), 4)
        if not args.skip_f0:
            report["f0_source_vs_converted"] = compare_f0(args.source, args.converted, args.semitone_shift,
                                                         args.length_adjust, args.auto_f0_adjust)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
