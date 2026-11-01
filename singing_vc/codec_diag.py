"""Numpy-only helpers for the DACVAE round-trip noise diagnosis.

Gain conditioning (boost before encode, exact inverse after decode), delay /
gain alignment and spectral bookkeeping, kept model-free so they can be unit
tested. Input-output error mixes phase, distortion and added energy; only the
``quiet_*`` and ``added_*`` numbers isolate energy the codec puts where the
source had little, and even those are proxies next to listening.
"""

from __future__ import annotations

import math

import numpy as np

from singing_vc import eval_metrics as em

BANDS_HZ = ((0, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 16000), (16000, 22000),
            (22050, 24000))  # last band: above a 44.1 kHz source's Nyquist, so anything there is generated
CONTRAST_BANDS_HZ = ((300, 3000), (3000, 8000), (8000, 16000))
QUIET_DBFS = -55.0  # source frames below this (and far below the loudest frame) count as near-silent
QUIET_RELATIVE_DB = 40.0
N_FFT = 2048
HOP = 512
EPS = 1e-10


def peak_safe_gain_db(audio: np.ndarray, wanted_db: float, peak_cap: float = 0.95) -> dict:
    """``wanted_db`` limited so the conditioned peak never exceeds ``peak_cap`` (attenuates if already above)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    headroom = 20.0 * math.log10(peak_cap / peak) if peak > 0 else math.inf
    gain = min(wanted_db, headroom)
    return {"wanted_db": wanted_db, "gain_db": gain, "peak_limited": bool(gain < wanted_db),
            "input_peak": peak, "conditioned_peak": peak * 10 ** (gain / 20)}


def scalar_gain_db(reference: np.ndarray, scaled: np.ndarray) -> float:
    """dB of the least-squares scalar mapping ``reference`` onto ``scaled`` (exact for a pure gain)."""
    ref = np.asarray(reference, dtype=np.float64)
    out = np.asarray(scaled, dtype=np.float64)[: len(ref)]
    ratio = float(np.dot(ref[: len(out)], out) / max(np.dot(ref[: len(out)], ref[: len(out)]), EPS))
    return 20.0 * math.log10(ratio) if ratio > 0 else -math.inf


def restore_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (np.asarray(audio, dtype=np.float64) * 10 ** (-gain_db / 20)).astype(np.float32)


def best_lag(reference: np.ndarray, test: np.ndarray, max_lag: int) -> int:
    """Lag (samples) maximising cross-correlation; positive = ``test`` is late."""
    n = len(reference) + len(test)
    size = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(test, size) * np.conj(np.fft.rfft(reference, size))
    corr = np.fft.irfft(spec, size)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    values = np.concatenate([corr[: max_lag + 1], corr[size - max_lag:]])
    return int(lags[int(np.argmax(values))])


def align(reference: np.ndarray, test: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift ``test`` back by ``lag`` and crop both to the common span."""
    if lag > 0:
        test = test[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    n = min(len(reference), len(test))
    return np.asarray(reference[:n], dtype=np.float64), np.asarray(test[:n], dtype=np.float64)


def power_spectrogram(audio: np.ndarray) -> np.ndarray:
    """(frames, bins) power with a Hann window; frame i is centred at i * HOP."""
    audio = np.pad(np.asarray(audio, dtype=np.float64), (N_FFT // 2, N_FFT // 2))
    count = 1 + (len(audio) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(count)[:, None]
    window = np.hanning(N_FFT)
    return np.abs(np.fft.rfft(audio[index] * window, axis=1)) ** 2 / np.sum(window ** 2)


def frame_levels_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power.sum(axis=1) * 2 / N_FFT, EPS))


def frame_masks(ref_levels_db: np.ndarray) -> dict[str, np.ndarray]:
    """Source-defined frame classes: active (singing/speech), quiet (near-silent), and the rest."""
    active = em.active_mask(ref_levels_db)
    top = float(np.max(ref_levels_db)) if ref_levels_db.size else 0.0
    quiet = (ref_levels_db < QUIET_DBFS) & (ref_levels_db < top - QUIET_RELATIVE_DB)
    return {"active": active, "quiet": quiet, "low": ~active & ~quiet}


def band_levels_db(power: np.ndarray, mask: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    out = {}
    for lo, hi in BANDS_HZ:
        if lo >= sample_rate / 2:
            continue
        cols = (freqs >= lo) & (freqs < hi)
        if not mask.any() or not cols.any():
            out[f"{lo}-{hi}"] = None
            continue
        out[f"{lo}-{hi}"] = 10.0 * math.log10(max(float(power[mask][:, cols].sum(axis=1).mean()), EPS))
    return out


def spectral_contrast_db(power: np.ndarray, mask: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    """Mean per-frame (90th - 25th percentile) bin level per band: harmonic peaks over the floor between them.

    Computed on each signal independently, so unlike source-selected "valley" bins it has no
    regression-to-the-mean bias. A drop means energy fills in between partials (or partials flatten).
    """
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    out = {}
    for lo, hi in CONTRAST_BANDS_HZ:
        cols = (freqs >= lo) & (freqs < hi)
        if not mask.any() or not cols.any():
            out[f"{lo}-{hi}"] = None
            continue
        level = 10 * np.log10(power[mask][:, cols] + EPS)
        out[f"{lo}-{hi}"] = float(np.mean(np.percentile(level, 90, axis=1) - np.percentile(level, 25, axis=1)))
    return out


def compare(reference: np.ndarray, decoded: np.ndarray, sample_rate: int, max_lag: int = 2400) -> dict:
    """Reconstruction properties of ``decoded`` against ``reference`` (same rate, same level domain)."""
    ref = np.asarray(reference, dtype=np.float64)
    dec = np.asarray(decoded, dtype=np.float64)
    lag = best_lag(ref, dec, max_lag)
    r, d = align(ref, dec, lag)
    gain_db = scalar_gain_db(r, d)
    g = 10 ** (gain_db / 20) if math.isfinite(gain_db) else 1.0
    err = d - g * r
    pr, pd = power_spectrogram(r), power_spectrogram(d)
    n = min(len(pr), len(pd))
    pr, pd = pr[:n], pd[:n]
    ref_db, dec_db = frame_levels_db(pr), frame_levels_db(pd)
    masks = frame_masks(ref_db)
    lr, ld = 10 * np.log10(pr + EPS), 10 * np.log10(pd + EPS)
    floor = np.max(lr) - 80.0  # ignore bins far below anything audible in the reference
    keep = np.maximum(lr, ld) > floor
    diff = np.where(keep, ld - lr, 0.0)
    result = {
        "lag_samples": lag, "decoded_samples": len(dec), "reference_samples": len(ref),
        "fit_gain_db": gain_db,
        "rms_dbfs": em.db(float(np.sqrt(np.mean(d ** 2)))) if d.size else None,
        "reference_rms_dbfs": em.db(float(np.sqrt(np.mean(r ** 2)))) if r.size else None,
        "peak": float(np.max(np.abs(dec))) if dec.size else 0.0,
        "clipped_ratio": float(np.mean(np.abs(dec) >= 0.999)) if dec.size else 0.0,
        "finite": bool(np.isfinite(dec).all()),
        # Mixes phase error, distortion and added energy: not a noise measure by itself.
        "waveform_error_ratio_db": 10 * math.log10(max(float(np.sum(err ** 2)), EPS) / max(float(np.sum(r ** 2)), EPS)),
        "frames": {k: int(v.sum()) for k, v in masks.items()},
    }
    for name, mask in masks.items():
        if not mask.any():
            result[name] = None
            continue
        m = mask[:, None] & keep
        positive = np.maximum(diff, 0.0)
        result[name] = {
            "reference_level_db": float(np.median(ref_db[mask])),
            "decoded_level_db": float(np.median(dec_db[mask])),
            "level_excess_db": float(np.median(dec_db[mask] - ref_db[mask])),
            # Floor relative to each signal's own active level, so a quieter decode is not read as less noise.
            "reference_level_rel_active_db": float(np.median(ref_db[mask]) - np.median(ref_db[masks["active"]]))
            if masks["active"].any() else None,
            "decoded_level_rel_active_db": float(np.median(dec_db[mask]) - np.median(dec_db[masks["active"]]))
            if masks["active"].any() else None,
            "log_spectral_distance_db": float(np.sqrt(np.mean(diff[mask] ** 2))),
            "mean_bin_diff_db": float(diff[mask][m[mask]].mean()) if m[mask].any() else None,
            # Mean of the positive part of per-bin dB differences: where the decode is louder than the
            # source, in dB averaged over bins. Includes smearing/phase-window effects; not a noise power.
            "mean_positive_bin_diff_db": float(positive[mask].mean()),
            "reference_contrast_db": spectral_contrast_db(pr, mask, sample_rate),
            "decoded_contrast_db": spectral_contrast_db(pd, mask, sample_rate),
            "reference_bands_db": band_levels_db(pr, mask, sample_rate),
            "decoded_bands_db": band_levels_db(pd, mask, sample_rate),
        }
    return result


def synthetic_controls(sample_rate: int) -> dict[str, np.ndarray]:
    """Out-of-domain probes: digital silence, and a harmonic tone with silent gaps at two levels."""
    t = np.arange(sample_rate) / sample_rate
    harmonic = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 6))
    harmonic = harmonic / np.max(np.abs(harmonic))
    gap = np.zeros(sample_rate)
    tone_gaps = np.concatenate([gap, 0.3 * harmonic, gap, 0.03 * harmonic, gap])
    return {"silence_4s": np.zeros(4 * sample_rate, dtype=np.float32), "tone_gaps": tone_gaps.astype(np.float32)}
