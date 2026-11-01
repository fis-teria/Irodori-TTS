"""Numpy-only helpers for the controlled reference-comparison experiment.

Everything here is deterministic signal bookkeeping (activity masks, window
selection, gain matching, F0 coverage, envelope lag, embedding aggregation) so
it can be unit-tested without models. None of these numbers measure perceived
voice likeness or singing quality; they are proxies to be read next to a
listening test.
"""

from __future__ import annotations

import math

import numpy as np

HOP_SECONDS = 0.010
FRAME_SECONDS = 0.025
ACTIVE_FLOOR_DBFS = -50.0
ACTIVE_RELATIVE_DB = 35.0


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else -math.inf


def frame_rms_db(audio: np.ndarray, sample_rate: int, frame_seconds: float = FRAME_SECONDS,
                 hop_seconds: float = HOP_SECONDS) -> np.ndarray:
    """RMS level (dBFS) of centred frames, one per hop; length = ceil(n / hop)."""
    mono = to_mono(audio)
    hop = max(1, int(round(hop_seconds * sample_rate)))
    frame = max(hop, int(round(frame_seconds * sample_rate)))
    count = int(math.ceil(len(mono) / hop)) if len(mono) else 0
    padded = np.pad(np.square(mono), (frame // 2, frame))
    cumulative = np.concatenate([[0.0], np.cumsum(padded)])
    starts = np.arange(count) * hop
    energy = (cumulative[starts + frame] - cumulative[starts]) / frame
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(energy, 1e-20))


def active_mask(levels_db: np.ndarray, floor_db: float = ACTIVE_FLOOR_DBFS,
                relative_db: float = ACTIVE_RELATIVE_DB) -> np.ndarray:
    """Frames louder than both an absolute floor and (loudest frame - relative_db)."""
    if levels_db.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = max(floor_db, float(np.max(levels_db)) - relative_db)
    return levels_db > threshold


def active_rms_db(audio: np.ndarray, sample_rate: int) -> float:
    """Power-average level of active frames only, so silence fraction does not shift the level."""
    levels = frame_rms_db(audio, sample_rate)
    mask = active_mask(levels)
    if not mask.any():
        return -math.inf
    return float(10.0 * np.log10(np.mean(10.0 ** (levels[mask] / 10.0))))


def best_window_start(mask: np.ndarray, window_frames: int) -> int:
    """First frame index of the contiguous window with the most active frames (earliest on ties)."""
    if window_frames >= len(mask):
        return 0
    cumulative = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    counts = cumulative[window_frames:] - cumulative[:-window_frames]
    return int(np.argmax(counts))


def matched_gain(level_db: float, peak: float, target_db: float = -24.0,
                 peak_ceiling_db: float = -1.0) -> dict:
    """Gain that brings ``level_db`` to ``target_db`` unless the peak would pass the ceiling."""
    wanted = target_db - level_db
    peak_db = db(peak)
    headroom = peak_ceiling_db - peak_db if math.isfinite(peak_db) else math.inf
    gain = min(wanted, headroom)
    return {"gain_db": gain, "peak_limited": bool(headroom < wanted),
            "achieved_level_db": level_db + gain, "achieved_peak_db": peak_db + gain}


def attenuation_only_gains(levels_db: list[float], peaks: list[float],
                           peak_ceiling_db: float = -1.0) -> tuple[float, list[float]]:
    """Common level every item can reach without boosting or exceeding the peak ceiling."""
    reachable = []
    for level, peak in zip(levels_db, peaks):
        peak_room = peak_ceiling_db - db(peak) if peak > 0 else math.inf
        reachable.append(level + min(0.0, peak_room))
    common = min(reachable)
    return common, [common - level for level in levels_db]


def select_windows(mask: np.ndarray, window_frames: int, min_active: float = 0.6) -> list[tuple[int, int]]:
    """Greedy left-to-right non-overlapping windows whose active fraction is >= ``min_active``."""
    windows = []
    if window_frames <= 0 or len(mask) < window_frames:
        return windows
    cumulative = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    start = 0
    need = min_active * window_frames
    while start + window_frames <= len(mask):
        if cumulative[start + window_frames] - cumulative[start] >= need:
            windows.append((start, start + window_frames))
            start += window_frames
        else:
            start += 1
    return windows


def unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of L2-normalised window embeddings, renormalised to unit length."""
    return unit(unit(embeddings).mean(axis=0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def f0_coverage(src: np.ndarray, out: np.ndarray, semitone_shift: float = 0.0) -> dict:
    """F0 agreement with every source-voiced frame in the denominator.

    A source-voiced frame where the output is unvoiced counts as a failure, so
    conditions with different output voicing are compared on the same base.
    The common-voiced subset is reported separately and labelled as such.
    """
    frames = min(len(src), len(out))
    src, out = np.asarray(src[:frames], float), np.asarray(out[:frames], float)
    src_voiced = ~np.isnan(src)
    both = src_voiced & ~np.isnan(out)
    n_src = int(src_voiced.sum())
    result = {"frames": frames, "source_voiced_frames": n_src, "common_voiced_frames": int(both.sum()),
              "output_voiced_frames": int((~np.isnan(out)).sum())}
    if n_src == 0:
        return result
    cents = np.full(frames, np.nan)
    cents[both] = 1200.0 * (np.log2(out[both]) - np.log2(src[both])) - 100.0 * semitone_shift
    within = both & (np.abs(np.nan_to_num(cents, nan=np.inf)) <= 50.0)
    result.update({
        "output_valid_on_source_voiced": both.sum() / n_src,
        "within_50c_of_source_voiced": within.sum() / n_src,
        "octave_error_of_source_voiced": (both & (np.abs(np.nan_to_num(cents)) >= 600.0)).sum() / n_src,
    })
    if both.sum() >= 10:
        log_src, log_out = np.log2(src[both]), np.log2(out[both])
        flat = np.ptp(log_src) == 0 or np.ptp(log_out) == 0
        result.update({
            "common_median_abs_cents": float(np.median(np.abs(cents[both]))),
            "common_within_50c": float(within.sum() / both.sum()),
            "common_log_f0_correlation": None if flat else float(np.corrcoef(log_src, log_out)[0, 1]),
        })
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in result.items()}


def envelope_lag(reference_db: np.ndarray, test_db: np.ndarray, max_lag: int) -> dict:
    """Lag (frames; positive = test is late) maximising Pearson correlation of two level envelopes.

    A coarse timing proxy only: it cannot see phoneme or note-onset alignment.
    """
    frames = min(len(reference_db), len(test_db))
    a, b = np.asarray(reference_db[:frames], float), np.asarray(test_db[:frames], float)

    def corr(lag: int) -> float:
        x, y = (a[:frames - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:frames + lag])
        if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return -math.inf
        return float(np.corrcoef(x, y)[0, 1])

    scores = {lag: corr(lag) for lag in range(-max_lag, max_lag + 1)}
    best = max(scores, key=lambda lag: (scores[lag], -abs(lag)))
    return {"best_lag_frames": int(best), "corr_at_best": scores[best], "corr_at_zero": scores[0],
            "max_lag_frames": int(max_lag)}


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Levenshtein distance over characters (whitespace removed) / reference length."""
    ref = "".join(reference.split())
    hyp = "".join(hypothesis.split())
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1] / len(ref)


def blind_labels(keys: list[str], seed: int) -> dict[str, str]:
    """Deterministic shuffled mapping key -> neutral label (X01, X02, ...)."""
    order = list(keys)
    np.random.default_rng(seed).shuffle(order)
    return {key: f"X{i + 1:02d}" for i, key in enumerate(order)}
