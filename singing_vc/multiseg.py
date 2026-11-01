"""Model-free helpers for the multi-segment DACVAE codec validation.

Interval planning with exact sample boundaries, a pre-registered "essentially silent"
check plus a deterministic replacement rule, source-defined quiet spans for DC/AC
analysis, crop/padding bookkeeping, and seeded A/B label assignment for a listening
pack. Nothing here scores quality; these are bookkeeping rules fixed before inference.
"""

from __future__ import annotations

import math

import numpy as np

from singing_vc import eval_metrics as em

# Pre-registered activity rule (fixed before any codec inference).
SILENT_RMS_DBFS = -50.0         # whole-interval mono RMS below this -> essentially silent
SILENT_ACTIVE_FRACTION = 0.25   # or fewer than this fraction of 25 ms frames above -50 dBFS
ACTIVE_ABS_FLOOR_DBFS = em.ACTIVE_FLOOR_DBFS
ALTERNATIVE_STEP_SECONDS = 5.0

# Source-defined quiet span for DC/AC analysis.
QUIET_SPAN_DBFS = -60.0
QUIET_SPAN_MIN_SECONDS = 0.2
QUIET_SPAN_ERODE_SECONDS = 0.05


def interval_samples(start_s: float, end_s: float, sample_rate: int, total_frames: int) -> tuple[int, int]:
    """Exact [first, last) sample indices; refuses intervals that do not fit the file."""
    if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s <= start_s or start_s < 0:
        raise ValueError(f"invalid interval {start_s}-{end_s}")
    first, last = int(round(start_s * sample_rate)), int(round(end_s * sample_rate))
    if last > total_frames:
        raise ValueError(f"interval {start_s}-{end_s}s ends after the file ({total_frames / sample_rate:.3f}s)")
    return first, last


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def activity(audio: np.ndarray, sample_rate: int) -> dict:
    """Level facts for one source interval (channel mean) and the pre-registered silence verdict."""
    mono = em.to_mono(audio)
    if not mono.size or not np.isfinite(mono).all():
        raise ValueError("activity needs nonempty finite samples")
    levels = em.frame_rms_db(mono, sample_rate)
    rms = float(np.sqrt(np.mean(mono * mono)))
    rms_db = em.db(rms)
    abs_fraction = float(np.mean(levels > ACTIVE_ABS_FLOOR_DBFS))
    silent = bool(rms_db < SILENT_RMS_DBFS or abs_fraction < SILENT_ACTIVE_FRACTION)
    return {"rms_dbfs": rms_db if math.isfinite(rms_db) else None, "peak": float(np.max(np.abs(mono))),
            "frames_25ms": int(levels.size), "fraction_above_minus50dbfs": abs_fraction,
            "relative_active_fraction": float(np.mean(em.active_mask(levels))),
            "active_rms_db": em.active_rms_db(mono, sample_rate) if not silent else None,
            "essentially_silent": silent}


def alternative_start(silent_start_s: float, length_s: float, duration_s: float,
                      taken: list[tuple[float, float]], is_active) -> float | None:
    """First start >= the silent one (then wrapping from 0) on a 5 s grid that fits the file,
    overlaps no taken interval and passes ``is_active(start)``. Deterministic; uses source only."""
    step = ALTERNATIVE_STEP_SECONDS
    grid = [k * step for k in range(int(math.floor((duration_s - length_s) / step)) + 1)]
    ordered = [s for s in grid if s >= silent_start_s] + [s for s in grid if s < silent_start_s]
    for start in ordered:
        if any(start < b and a < start + length_s for a, b in taken):
            continue
        if is_active(start):
            return start
    return None


def longest_quiet_span(mono: np.ndarray, sample_rate: int) -> dict | None:
    """Longest source span that is exactly zero, else longest run of 10 ms frames below -60 dBFS.

    Eroded by 50 ms per side; ``None`` when nothing of at least 200 ms remains (metric unavailable).
    """
    x = np.asarray(mono, dtype=np.float64)
    erode = int(round(QUIET_SPAN_ERODE_SECONDS * sample_rate))
    need = int(round(QUIET_SPAN_MIN_SECONDS * sample_rate))

    def longest(flags: np.ndarray) -> tuple[int, int]:
        edges = np.diff(np.r_[0, flags.astype(np.int8), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if not starts.size:
            return 0, 0
        i = int(np.argmax(ends - starts))
        return int(starts[i]), int(ends[i])

    a, b = longest(x == 0)
    if b - a - 2 * erode >= need:
        return {"kind": "exact_zero", "start": a + erode, "end": b - erode}
    hop = int(round(em.HOP_SECONDS * sample_rate))
    levels = em.frame_rms_db(x, sample_rate)
    fa, fb = longest(levels < QUIET_SPAN_DBFS)
    a, b = fa * hop, min(len(x), fb * hop)
    if b - a - 2 * erode >= need:
        return {"kind": f"frames_below_{QUIET_SPAN_DBFS:g}dbfs", "start": a + erode, "end": b - erode}
    return None


def crop_to_reference(decoded: np.ndarray, reference_samples: int) -> tuple[np.ndarray, dict]:
    """Crop decoder tail padding to the reference length; never zero-pads a short decode."""
    n = len(decoded)
    if n < reference_samples:
        raise ValueError(f"decoded {n} samples < reference {reference_samples}; refusing to pad")
    return decoded[:reference_samples], {"decoded_samples": int(n), "reference_samples": int(reference_samples),
                                         "cropped_tail_samples": int(n - reference_samples), "zero_padded_samples": 0}


def shared_mask_levels(reference: np.ndarray, signals: list[np.ndarray], sample_rate: int) -> dict:
    """Power-average level of every signal over the frames the *reference* marks active.

    Unlike ``em.active_rms_db`` (each signal picks its own threshold), every signal is measured on
    one identical source-defined frame set, so level spreads are comparable. Signals are compared
    over the reference length; a shorter signal is an error.
    """
    ref_levels = em.frame_rms_db(reference, sample_rate)
    mask = em.active_mask(ref_levels)
    if not mask.any():
        raise ValueError("reference has no active frames")
    out = []
    for signal in signals:
        if len(signal) < len(reference):
            raise ValueError("signal shorter than the reference")
        levels = em.frame_rms_db(np.asarray(signal)[: len(reference)], sample_rate)
        out.append(float(10.0 * np.log10(np.mean(10.0 ** (levels[mask] / 10.0)))))
    return {"mask_frames": int(mask.sum()), "frames": int(mask.size), "levels_db": out}


def assign_ab(segments: list[str], seed: int, methods: tuple[str, str] = ("baseline", "candidate")) -> dict:
    """Seeded, per-segment independent assignment of two methods to labels A/B."""
    rng = np.random.default_rng(seed)
    out = {}
    for seg in segments:
        flip = bool(rng.integers(0, 2))
        out[seg] = {"A": methods[1] if flip else methods[0], "B": methods[0] if flip else methods[1]}
    return out
