"""Controlled comparison of Seed-VC singing references: generic Irodori (A), real target (B),
Irodori clone of the target (C).

Runs inside the Seed-VC venv (``.singing\\venv``); Irodori generation and the
DACVAE round trip run as ``.venv`` subprocesses and every Seed-VC conversion is
its own seeded subprocess, so only one model is GPU-resident at a time.

Stages (each reusable on an existing ``--root``):
  setup     decode target, split first part (conditioning) / rest (held-out), prepare source, audit old runs
  clone     Irodori v4.1-Small voice clone of the target from the conditioning part only
  refs      equal-length, equal-level references A/B/C (+ supplementary full-length B)
  vc        seeded Seed-VC conversions (conditions x seeds, plus one repeat for repeatability)
  cohort    optional same-speaker and impostor recordings for calibrating the identity proxy
  roundtrip DACVAE encode/decode of the source excerpt and held-out target speech
  eval      WavLM-SV identity proxies, pYIN F0 coverage, waveform facts, envelope lag, optional ASR
  summary   per-condition min/mean/max over VC seeds -> summary.json
  pack      loudness-matched blind listening pack (key stored separately)

All outputs stay under ``outputs/singing/reference_comparison_<timestamp>`` (git-ignored).
Numbers here are proxies; perceived likeness and singing quality need listening.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from singing_vc import audio_tool  # noqa: E402
from singing_vc import eval_metrics as em  # noqa: E402
from singing_vc.pipeline import (  # noqa: E402
    DEFAULT_HF_CHECKPOINT, IRODORI_PYTHON, SEEDVC_DIR, SEEDVC_PYTHON, child_environment,
    default_seedvc_model, git_revision, run_step,
)

SR = 44100
EMB_SR = 16000
EMB_WINDOW_SECONDS = 4.0
EMB_MIN_ACTIVE = 0.6
REF_LEVEL_DB = -24.0
PEAK_CEILING_DB = -1.0
ENVELOPE_MAX_LAG = 20  # frames of 10 ms
VC_SETTINGS = {"diffusion_steps": 30, "inference_cfg_rate": 0.7, "length_adjust": 1.0, "f0_condition": True,
               "auto_f0_adjust": False, "semitone_shift": 0, "fp16": True}
MAIN_CONDITIONS = ("A_generic", "B_real", "C_clone")
SUPPLEMENTARY_CONDITIONS = ("B_real_full",)
# Post-hoc, chosen after seeing pre-SVC identity scores: exploratory only, not part of the planned comparison.
EXPLORATORY_CONDITIONS = {"C_clone_ttsseed2": "C_clone_extra_seed2.wav"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=_jsonable),
                    encoding="utf-8")


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def clean(value):
    """Replace non-finite floats with None and round, recursively, for allow_nan=False JSON."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_mono(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate is not None and rate != sample_rate:
        import librosa

        mono = librosa.resample(mono, orig_sr=rate, target_sr=sample_rate, res_type="soxr_hq")
        rate = sample_rate
    return mono.astype(np.float32), rate


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- setup


def stage_setup(root: Path, args: argparse.Namespace) -> None:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    target_full = inputs / "target_full.wav"
    if not target_full.exists():
        subprocess.run([str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i",
                        str(args.target_audio), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(target_full)],
                       check=True)
    audio, rate = sf.read(str(target_full), dtype="int16")
    cut = int(round(args.conditioning_seconds * rate))
    sf.write(str(inputs / "target_conditioning.wav"), audio[:cut], rate, subtype="PCM_16")
    sf.write(str(inputs / "target_heldout.wav"), audio[cut:], rate, subtype="PCM_16")
    audio_tool.prepare(args.source_audio, inputs / "source_excerpt.wav", args.source_start, args.source_duration)
    shutil.copyfile(args.generic_ref, inputs / "A_generic_original.wav")

    old = {}
    for name, run_dir in (("A_generic_old_20s", args.old_a_run), ("B_real25_old_20s", args.old_b_run)):
        meta = read_json(run_dir / "metadata.json")
        analysis = read_json(run_dir / "analysis.json")
        old[name] = {
            "run_dir_name": run_dir.name,
            "reference_mode": meta.get("reference_mode"),
            "irodori_revision": meta.get("revisions", {}).get("irodori_checkout"),
            "options": {k: meta["options"].get(k) for k in ("source_start", "source_duration", "diffusion_steps",
                        "inference_cfg_rate", "length_adjust", "f0_condition", "auto_f0_adjust",
                        "semitone_shift", "fp16", "tts_hf_checkpoint", "tts_seed", "tts_speaker_wav")},
            "tts_caption_present": bool(meta["options"].get("tts_caption")),
            "reference": {k: analysis["reference"].get(k) for k in ("sample_rate", "duration_seconds", "rms_dbfs", "peak")},
            "converted": {k: analysis["converted"].get(k) for k in ("duration_seconds", "rms_dbfs", "peak")},
            "f0_reported": analysis.get("f0_source_vs_converted"),
            "source_sha256": sha256(run_dir / "source.wav"),
            "reference_sha256": sha256(run_dir / "reference.wav"),
            "converted_sha256": sha256(run_dir / "converted.wav"),
            "vc_seed_recorded": False,
        }
        shutil.copyfile(run_dir / "converted.wav", inputs / f"{name}_converted.wav")
    excerpt_hash = sha256(inputs / "source_excerpt.wav")
    setup = {
        "target_original_sha256": sha256(args.target_audio),
        "source_original_sha256": sha256(args.source_audio),
        "target_total_seconds": round(len(audio) / rate, 3),
        "conditioning_seconds": [0.0, args.conditioning_seconds],
        "heldout_seconds": [args.conditioning_seconds, round(len(audio) / rate, 3)],
        "source_excerpt_seconds": [args.source_start, args.source_start + args.source_duration],
        "source_excerpt_sha256": excerpt_hash,
        "source_excerpt_matches_old_runs": all(o["source_sha256"] == excerpt_hash for o in old.values()),
        "old_b_reference_matches_new_conditioning_samples":
            _same_samples(args.old_b_reference, inputs / "target_conditioning.wav") if args.old_b_reference else None,
        "old_runs": old,
        "observational_only": "old runs have different reference identities, lengths and levels and no recorded VC seed",
    }
    write_json(root / "setup.json", clean(setup))
    log(f"[setup] target {setup['target_total_seconds']}s, source excerpt hash match {setup['source_excerpt_matches_old_runs']}")


def _same_samples(a: Path, b: Path) -> bool:
    x, _ = sf.read(str(a), dtype="int16")
    y, _ = sf.read(str(b), dtype="int16")
    return x.shape == y.shape and bool(np.array_equal(x, y))


# --------------------------------------------------------------------------- clone


def stage_clone(root: Path, args: argparse.Namespace) -> None:
    """Clone with the main seed (used for SVC) plus optional extra seeds (pre-SVC variability only)."""
    text = args.clone_text or read_json(args.old_a_run / "metadata.json")["options"]["tts_text"]
    env = child_environment(True)
    report = {"checkpoint": args.tts_hf_checkpoint, "caption": None,
              "text_same_as_generic_reference": args.clone_text is None,
              "speaker_reference": "target conditioning part only (held-out part unused)", "outputs": {}}
    for seed in [args.tts_seed] + list(args.extra_clone_seeds):
        main = seed == args.tts_seed
        out = root / "inputs" / ("C_clone_original.wav" if main else f"C_clone_extra_seed{seed}.wav")
        if out.exists():
            log(f"[clone] {out.name} exists, keeping it")
        else:
            command = [str(IRODORI_PYTHON), str(PROJECT_ROOT / "infer.py"), "--hf-checkpoint", args.tts_hf_checkpoint,
                       "--text", text, "--ref-wav", str(root / "inputs" / "target_conditioning.wav"),
                       "--seed", str(seed), "--output-wav", str(out)]
            run_step(f"irodori_clone_seed{seed}", command, root / "logs", env=env, log=log)
        report["outputs"][out.name] = {"seed": seed, "used_for_svc": main, "sha256": sha256(out),
                                       "seconds": sf.info(str(out)).duration}
    write_json(root / "clone.json", report)


# --------------------------------------------------------------------------- refs


def matched_reference(audio: np.ndarray, rate: int, seconds: float | None) -> tuple[np.ndarray, dict]:
    levels = em.frame_rms_db(audio, rate)
    mask = em.active_mask(levels)
    start = 0
    if seconds is not None:
        window = int(round(seconds / em.HOP_SECONDS))
        start_frame = em.best_window_start(mask, window)
        start = int(round(start_frame * em.HOP_SECONDS * rate))
        audio = audio[start:start + int(round(seconds * rate))]
    level = em.active_rms_db(audio, rate)
    gain = em.matched_gain(level, float(np.max(np.abs(audio))), REF_LEVEL_DB, PEAK_CEILING_DB)
    out = (audio * 10 ** (gain["gain_db"] / 20)).astype(np.float32)
    out_mask = em.active_mask(em.frame_rms_db(out, rate))
    facts = {"start_seconds": start / rate, "duration_seconds": len(out) / rate,
             "active_seconds": float(out_mask.sum() * em.HOP_SECONDS), "input_active_rms_db": level,
             "whole_rms_db_after": em.db(float(np.sqrt(np.mean(np.square(out, dtype=np.float64))))), **gain}
    return out, facts


def stage_refs(root: Path, args: argparse.Namespace) -> None:
    inputs, refs = root / "inputs", root / "refs"
    refs.mkdir(exist_ok=True)
    generic, _ = load_mono(inputs / "A_generic_original.wav", SR)
    length = len(generic) / SR
    sources = {"A_generic": (inputs / "A_generic_original.wav", length),
               "B_real": (inputs / "target_conditioning.wav", length),
               "C_clone": (inputs / "C_clone_original.wav", length),
               "B_real_full": (inputs / "target_conditioning.wav", None)}
    for name, file in EXPLORATORY_CONDITIONS.items():
        if (inputs / file).exists():
            sources[name] = (inputs / file, length)
    report = {"matched_length_seconds": length, "level_target_active_rms_db": REF_LEVEL_DB,
              "peak_ceiling_dbfs": PEAK_CEILING_DB, "sample_rate": SR, "conditions": {}}
    for name, (path, seconds) in sources.items():
        audio, _ = load_mono(path, SR)
        original_seconds = len(audio) / SR
        if seconds is not None and original_seconds < seconds - 1e-3:
            log(f"[refs] warning: {name} is {original_seconds:.2f}s, shorter than {seconds:.2f}s; using all of it")
            seconds = None
        out, facts = matched_reference(audio, SR, seconds)
        target = refs / f"{name}.wav"
        sf.write(str(target), out, SR, subtype="FLOAT")
        report["conditions"][name] = {"from": path.name, "original_seconds": original_seconds, **facts,
                                      "sha256": sha256(target)}
        log(f"[refs] {name}: {facts['duration_seconds']:.2f}s active {facts['active_seconds']:.2f}s "
            f"gain {facts['gain_db']:+.2f} dB peak-limited={facts['peak_limited']}")
    write_json(root / "refs.json", clean(report))


# --------------------------------------------------------------------------- vc


def vc_jobs(seeds: list[int], root: Path | None = None) -> list[dict]:
    jobs = [{"condition": c, "seed": s, "role": "main"} for c in MAIN_CONDITIONS for s in seeds]
    jobs += [{"condition": c, "seed": s, "role": "supplementary"} for c in SUPPLEMENTARY_CONDITIONS for s in seeds]
    if root is not None:
        jobs += [{"condition": c, "seed": s, "role": "exploratory"} for c in EXPLORATORY_CONDITIONS for s in seeds
                 if (root / "refs" / f"{c}.wav").exists()]
    jobs.append({"condition": "B_real", "seed": seeds[0], "role": "repeatability", "repeat": 1})
    for job in jobs:
        job["name"] = f"{job['condition']}_seed{job['seed']}" + (f"_repeat{job['repeat']}" if "repeat" in job else "")
    return jobs


def stage_vc(root: Path, args: argparse.Namespace) -> None:
    checkpoint, config = default_seedvc_model()
    env = child_environment(True)
    results = []
    for job in vc_jobs(args.seeds, root):
        run_dir = root / "vc" / job["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "converted.wav").exists() and not args.force:
            log(f"[vc] {job['name']} exists, skipping")
        else:
            runner = [
                "--seedvc-dir", str(SEEDVC_DIR), "--source", str(root / "inputs" / "source_excerpt.wav"),
                "--target", str(root / "refs" / f"{job['condition']}.wav"),
                "--output-wav", str(run_dir / "converted.wav"), "--report-json", str(run_dir / "seedvc_report.json"),
                "--checkpoint", str(Path(checkpoint).resolve()), "--config", str(Path(config).resolve()),
                "--diffusion-steps", str(VC_SETTINGS["diffusion_steps"]),
                "--length-adjust", repr(VC_SETTINGS["length_adjust"]),
                "--inference-cfg-rate", repr(VC_SETTINGS["inference_cfg_rate"]),
                "--f0-condition", "True", "--auto-f0-adjust", "False", "--semi-tone-shift", "0",
                "--fp16", "True", "--offline", "True",
            ]
            command = [str(SEEDVC_PYTHON), str(PROJECT_ROOT / "singing_vc" / "seeded_seedvc.py"),
                       "--seed", str(job["seed"]), "--"] + runner
            run_step(f"vc_{job['name']}", command, root / "logs", cwd=SEEDVC_DIR, env=env, log=log)
        report = read_json(run_dir / "seedvc_report.json")
        results.append({**job, "converted_sha256": sha256(run_dir / "converted.wav"),
                        "reference_sha256": sha256(root / "refs" / f"{job['condition']}.wav"),
                        "elapsed_seconds": report.get("elapsed_seconds"), "device": report.get("device"),
                        "cuda_peak_memory_mb": report.get("cuda_peak_memory_mb"), "seeding": report.get("seeding")})
    write_json(root / "vc.json", {"settings": VC_SETTINGS, "checkpoint": Path(checkpoint).name,
                                  "config": Path(config).name, "runs": results})


# --------------------------------------------------------------------------- cohort


def stage_cohort(root: Path, args: argparse.Namespace) -> None:
    """Decode extra same-speaker recordings and other-speaker (impostor) recordings for calibration.

    Files are renamed to neutral indices; the index -> original path map stays in ``private/``.
    """
    folder = root / "inputs" / "cohort"
    folder.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for prefix, paths in (("same", args.same_speaker), ("impostor", args.cohort)):
        for index, path in enumerate(paths or []):
            name = f"{prefix}_{index:02d}"
            out = folder / f"{name}.wav"
            if not out.exists():
                subprocess.run([str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i",
                                str(path), "-t", str(args.cohort_seconds), "-ac", "1", "-ar", str(SR),
                                "-c:a", "pcm_s16le", str(out)], check=True)
            mapping[name] = {"source": str(path), "sha256_original": sha256(path), "seconds": sf.info(str(out)).duration}
    write_json(root / "private" / "cohort_sources.json", mapping)
    log(f"[cohort] {sum(n.startswith('same') for n in mapping)} same-speaker, "
        f"{sum(n.startswith('impostor') for n in mapping)} impostor recordings")


# --------------------------------------------------------------------------- roundtrip


def stage_roundtrip(root: Path, args: argparse.Namespace) -> None:
    rt = root / "roundtrip"
    inputs = [root / "inputs" / "source_excerpt.wav", root / "inputs" / "target_heldout.wav"]
    outputs = [rt / "source_excerpt_dacvae.wav", rt / "target_heldout_dacvae.wav"]
    command = [str(IRODORI_PYTHON), str(PROJECT_ROOT / "scripts" / "dacvae_roundtrip.py"),
               "--input", *map(str, inputs), "--output", *map(str, outputs), "--json", str(rt / "roundtrip.json")]
    run_step("dacvae_roundtrip", command, root / "logs", env=child_environment(True), log=log)


# --------------------------------------------------------------------------- eval


def audio_items(root: Path, args: argparse.Namespace) -> dict[str, dict]:
    """name -> {"path", "kind", optional "segments" in seconds}."""
    inputs = root / "inputs"
    source_total = sf.info(str(args.source_audio)).duration
    items = {
        "target_heldout": {"path": inputs / "target_heldout.wav", "kind": "anchor"},
        "target_conditioning": {"path": inputs / "target_conditioning.wav", "kind": "anchor"},
        "source_excerpt": {"path": inputs / "source_excerpt.wav", "kind": "anchor"},
        "source_singer_other": {"path": args.source_audio, "kind": "anchor",
                                "segments": [(0.0, args.source_start),
                                             (args.source_start + args.source_duration, source_total)]},
        "A_generic_original": {"path": inputs / "A_generic_original.wav", "kind": "reference_original"},
        "C_clone_original": {"path": inputs / "C_clone_original.wav", "kind": "reference_original"},
        **{p.stem: {"path": p, "kind": "reference_original"} for p in sorted(inputs.glob("C_clone_extra_seed*.wav"))},
        "A_generic_old_20s": {"path": inputs / "A_generic_old_20s_converted.wav", "kind": "old_output"},
        "B_real25_old_20s": {"path": inputs / "B_real25_old_20s_converted.wav", "kind": "old_output"},
    }
    for name in MAIN_CONDITIONS + SUPPLEMENTARY_CONDITIONS + tuple(EXPLORATORY_CONDITIONS):
        if (root / "refs" / f"{name}.wav").exists():
            items[f"ref_{name}"] = {"path": root / "refs" / f"{name}.wav", "kind": "reference_matched",
                                    "condition": name}
    for job in vc_jobs(args.seeds, root):
        items[f"out_{job['name']}"] = {"path": root / "vc" / job["name"] / "converted.wav", "kind": "output",
                                       "condition": job["condition"], "seed": job["seed"], "role": job["role"]}
    for name in ("source_excerpt", "target_heldout"):
        path = root / "roundtrip" / f"{name}_dacvae.wav"
        if path.exists():
            items[f"rt_{name}"] = {"path": path, "kind": "codec_roundtrip", "of": name}
    for path in sorted((root / "inputs" / "cohort").glob("*.wav")):
        items[path.stem] = {"path": path, "kind": "same_speaker" if path.stem.startswith("same") else "impostor"}
    return items


def embedding_windows(audio16: np.ndarray, segments_samples: list[tuple[int, int]]) -> list[np.ndarray]:
    windows = []
    frames = int(round(EMB_WINDOW_SECONDS / em.HOP_SECONDS))
    hop = int(round(em.HOP_SECONDS * EMB_SR))
    for first, last in segments_samples:
        part = audio16[first:last]
        mask = em.active_mask(em.frame_rms_db(part, EMB_SR))
        for start, end in em.select_windows(mask, frames, EMB_MIN_ACTIVE):
            windows.append(part[start * hop:end * hop])
    return windows


def wavlm_embeddings(items: dict[str, dict], model_dir: Path, device: str) -> dict[str, np.ndarray]:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    model = WavLMForXVector.from_pretrained(str(model_dir), local_files_only=True).eval().to(device)
    out = {}
    for name, item in items.items():
        audio, _ = load_mono(Path(item["path"]), EMB_SR)
        segments = [(int(a * EMB_SR), int(b * EMB_SR)) for a, b in item.get("segments", [(0, len(audio) / EMB_SR)])]
        windows = embedding_windows(audio, segments)
        if not windows:
            log(f"[wavlm] {name}: no qualifying windows")
            continue
        vectors = []
        with torch.inference_mode():
            for window in windows:
                features = extractor(window, sampling_rate=EMB_SR, return_tensors="pt").to(device)
                vectors.append(model(**features).embeddings[0].float().cpu().numpy())
        out[name] = np.stack(vectors)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def identity_table(embeddings: dict[str, np.ndarray], items: dict[str, dict]) -> dict:
    cents = {name: em.centroid(vectors) for name, vectors in embeddings.items()}
    anchors = {"heldout_target": "target_heldout", "source_singer_other_sections": "source_singer_other",
               "source_excerpt_same_section": "source_excerpt"}
    table = {}
    for name, vectors in embeddings.items():
        row = {"windows": len(vectors)}
        for label, anchor in anchors.items():
            if anchor in cents and anchor != name:
                row[f"cos_{label}"] = em.cosine(cents[name], cents[anchor])
        windows_vs_target = em.unit(vectors) @ cents["target_heldout"] if name != "target_heldout" else None
        if windows_vs_target is not None:
            row["window_cos_heldout_mean"] = float(windows_vs_target.mean())
            row["window_cos_heldout_std"] = float(windows_vs_target.std())
        condition = items[name].get("condition")
        if items[name]["kind"] == "output" and f"ref_{condition}" in cents:
            row["cos_own_condition_reference"] = em.cosine(cents[name], cents[f"ref_{condition}"])
        if "cos_heldout_target" in row and "cos_source_singer_other_sections" in row:
            row["target_minus_source_singer"] = row["cos_heldout_target"] - row["cos_source_singer_other_sections"]
        table[name] = row
    # Within-target split-half: held-out windows split into halves, a same-speaker same-domain ceiling.
    held = embeddings["target_heldout"]
    if len(held) >= 2:
        half = len(held) // 2
        table["target_heldout"]["split_half_cos"] = em.cosine(em.centroid(held[:half]), em.centroid(held[half:]))
    refs = [n for n in cents if n.startswith("ref_") or n.endswith("_original")]
    cross = {a: {b: em.cosine(cents[a], cents[b]) for b in refs if b != a} for a in refs}
    impostors = [n for n in cents if items[n]["kind"] == "impostor"]
    if impostors:
        # Closed-set check: is the held-out target nearer than every impostor speaker and the source singer?
        candidates = ["target_heldout", "source_singer_other"] + impostors
        for name, row in table.items():
            if name in candidates:
                continue
            scores = {c: em.cosine(cents[name], cents[c]) for c in candidates}
            others = np.array([scores[c] for c in impostors])
            row["target_rank_among_candidates"] = 1 + sum(scores[c] > scores["target_heldout"] for c in candidates)
            row["candidates"] = len(candidates)
            row["impostor_cos_mean"] = float(others.mean())
            row["impostor_cos_max"] = float(others.max())
            row["target_z_vs_impostors"] = float((scores["target_heldout"] - others.mean()) / max(others.std(), 1e-9))
    return {"per_item": table, "cross_reference": cross}


def f0_cached(path: Path, cache: Path) -> np.ndarray:
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / f"{sha256(path)[:20]}.npy"
    if file.exists():
        return np.load(file)
    contour = audio_tool.f0_contour(path)
    np.save(file, contour)
    return contour


def asr_transcripts(items: dict[str, dict], whisper_dir: Path, device: str) -> dict[str, str]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(str(whisper_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(whisper_dir), local_files_only=True).eval().to(device)
    texts = {}
    for name, item in items.items():
        audio, _ = load_mono(Path(item["path"]), EMB_SR)
        features = processor(audio, sampling_rate=EMB_SR, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            ids = model.generate(features, language="ja", task="transcribe", num_beams=1, do_sample=False)
        texts[name] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return texts


def stage_eval(root: Path, args: argparse.Namespace) -> None:
    items = audio_items(root, args)
    missing = [name for name, item in items.items() if not Path(item["path"]).exists()]
    if missing:
        raise SystemExit(f"missing audio for eval: {missing}")
    device = "cuda" if args.device == "cuda" else "cpu"

    log("[eval] WavLM-SV embeddings")
    embeddings = wavlm_embeddings(items, args.wavlm_dir, device)
    np.savez(root / "eval_embeddings.npz", **embeddings)
    identity = identity_table(embeddings, items)

    log("[eval] F0 / waveform / envelope")
    source = root / "inputs" / "source_excerpt.wav"
    src_f0 = f0_cached(source, root / "cache" / "f0")
    src_audio, _ = load_mono(source, SR)
    src_env = em.frame_rms_db(src_audio, SR)
    signal = {}
    for name, item in items.items():
        if item["kind"] not in ("output", "old_output", "codec_roundtrip") or item.get("of") == "target_heldout":
            continue
        path = Path(item["path"])
        audio, _ = load_mono(path, SR)
        signal[name] = {
            "waveform": {k: v for k, v in audio_tool.waveform_facts(path).items() if k != "path"},
            "f0_vs_source": em.f0_coverage(src_f0, f0_cached(path, root / "cache" / "f0")),
            "envelope_vs_source": em.envelope_lag(src_env, em.frame_rms_db(audio, SR), ENVELOPE_MAX_LAG),
        }
    rt_held = items.get("rt_target_heldout")
    if rt_held:
        held = root / "inputs" / "target_heldout.wav"
        signal["rt_target_heldout"] = {
            "waveform": {k: v for k, v in audio_tool.waveform_facts(Path(rt_held["path"])).items() if k != "path"},
            "f0_vs_original": em.f0_coverage(f0_cached(held, root / "cache" / "f0"),
                                             f0_cached(Path(rt_held["path"]), root / "cache" / "f0")),
        }
    for name, item in items.items():
        if item["kind"] == "codec_roundtrip":
            original, _ = load_mono(root / "inputs" / f"{item['of']}.wav", SR)
            decoded, _ = load_mono(Path(item["path"]), SR)
            n = min(len(original), len(decoded))
            err = decoded[:n] - original[:n]
            signal[name]["sample_snr_db"] = em.db(float(np.sqrt(np.mean(original[:n] ** 2)))) - em.db(
                float(np.sqrt(np.mean(err ** 2))))
            signal[name]["length_difference_samples"] = len(decoded) - len(original)

    asr = None
    if args.whisper_dir and args.whisper_dir.exists():
        log("[eval] Whisper-small ASR consistency (pseudo-reference = ASR of the source excerpt)")
        asr_items = {n: i for n, i in items.items()
                     if n == "source_excerpt" or i["kind"] in ("output", "old_output")
                     or n == "rt_source_excerpt"}
        texts = asr_transcripts(asr_items, args.whisper_dir, device)
        write_json(root / "private" / "asr_transcripts.json", texts)  # lyrics: never published
        asr = {n: {"cer_vs_source_asr": em.character_error_rate(texts["source_excerpt"], t),
                   "chars": len("".join(t.split()))} for n, t in texts.items() if n != "source_excerpt"}
        asr["_source_asr_chars"] = len("".join(texts["source_excerpt"].split()))
        tts_text = args.clone_text or read_json(args.old_a_run / "metadata.json")["options"]["tts_text"]
        speech = {n: i for n, i in items.items() if i["kind"] == "reference_original"}
        speech_texts = asr_transcripts(speech, args.whisper_dir, device)
        punctuation = str.maketrans("", "", "。、，．,.!?！？「」 ")
        asr["tts_speech_vs_input_text"] = {n: em.character_error_rate(tts_text.translate(punctuation),
                                                                      t.translate(punctuation))
                                           for n, t in speech_texts.items()}

    summary = {"items": {n: {k: v for k, v in i.items() if k not in ("path", "segments")} for n, i in items.items()},
               "item_sha256": {n: sha256(Path(i["path"])) for n, i in items.items()},
               "embedding": {"model": "microsoft/wavlm-base-plus-sv (local snapshot, read-only)",
                             "snapshot": args.wavlm_dir.name, "sample_rate": EMB_SR,
                             "window_seconds": EMB_WINDOW_SECONDS, "min_active_fraction": EMB_MIN_ACTIVE,
                             "aggregation": "L2-normalise each window x-vector, mean, renormalise; cosine of centroids",
                             "caveat": "uncalibrated speaker-verification proxy, trained on speech; not a likeness %"},
               "identity": identity, "signal": signal, "asr": asr,
               "f0_method": "librosa.pyin 16 kHz hop 10 ms 65-1100 Hz; source-voiced denominator",
               "envelope_method": f"10 ms frame dB envelopes, Pearson r, lag search +/-{ENVELOPE_MAX_LAG} frames"}
    write_json(root / "eval.json", clean(summary))
    log("[eval] wrote eval.json")


# --------------------------------------------------------------------------- pack


def stage_pack(root: Path, args: argparse.Namespace) -> None:
    pack, key_dir = root / "listening_pack", root / "listening_key"
    if pack.exists():
        raise FileExistsError(
            f"Listening pack already exists: {pack}. Keep its audio and ratings; use a fresh experiment root."
        )
    pack.mkdir()
    key_dir.mkdir(exist_ok=True)
    blind = {f"{c}_seed{s}": root / "vc" / f"{c}_seed{s}" / "converted.wav" for c in MAIN_CONDITIONS for s in args.seeds}
    anchors = {"anchor_target_voice_heldout": root / "inputs" / "target_heldout.wav",
               "anchor_source_song_excerpt": root / "inputs" / "source_excerpt.wav",
               "codec_gate_source_excerpt_dacvae": root / "roundtrip" / "source_excerpt_dacvae.wav"}
    anchors = {k: v for k, v in anchors.items() if v.exists()}
    audio = {name: load_mono(path, SR)[0] for name, path in {**blind, **anchors}.items()}
    if "anchor_target_voice_heldout" in audio:
        audio["anchor_target_voice_heldout"] = audio["anchor_target_voice_heldout"][:int(20 * SR)]
    names = list(audio)
    levels = [em.active_rms_db(audio[n], SR) for n in names]
    peaks = [float(np.max(np.abs(audio[n]))) for n in names]
    common, gains = em.attenuation_only_gains(levels, peaks, PEAK_CEILING_DB)
    labels = em.blind_labels(sorted(blind), args.pack_seed)
    key = {"common_active_rms_db": common, "peak_ceiling_dbfs": PEAK_CEILING_DB, "shuffle_seed": args.pack_seed,
           "labels": {}, "gains_db": {}}
    for name, gain in zip(names, gains):
        out = (audio[name] * 10 ** (gain / 20)).astype(np.float32)
        assert float(np.max(np.abs(out))) <= 10 ** (PEAK_CEILING_DB / 20) + 1e-6
        file = f"{labels[name]}.wav" if name in labels else f"{name}.wav"
        sf.write(str(pack / file), out, SR, subtype="PCM_16")
        key["gains_db"][file] = gain
        if name in labels:
            key["labels"][labels[name]] = name
    write_json(key_dir / "key.json", clean(key))
    shutil.copyfile(PROJECT_ROOT / "docs" / "SINGING_LISTENING_RUBRIC_JA.md", pack / "RUBRIC_JA.md")
    log(f"[pack] {len(labels)} blind items + {len(anchors)} anchors at {common:.1f} dB active RMS")


# --------------------------------------------------------------------------- summary

SUMMARY_METRICS = {
    "cos_heldout_target": ("identity", "cos_heldout_target"),
    "cos_source_singer_other": ("identity", "cos_source_singer_other_sections"),
    "target_z_vs_impostors": ("identity", "target_z_vs_impostors"),
    "target_rank": ("identity", "target_rank_among_candidates"),
    "cos_own_reference": ("identity", "cos_own_condition_reference"),
    "f0_valid_of_source_voiced": ("f0", "output_valid_on_source_voiced"),
    "f0_within50c_of_source_voiced": ("f0", "within_50c_of_source_voiced"),
    "f0_common_within50c": ("f0", "common_within_50c"),
    "f0_common_median_cents": ("f0", "common_median_abs_cents"),
    "f0_common_corr": ("f0", "common_log_f0_correlation"),
    "envelope_corr_zero_lag": ("env", "corr_at_zero"),
    "envelope_best_lag_frames": ("env", "best_lag_frames"),
    "rms_dbfs": ("wave", "rms_dbfs"),
    "peak": ("wave", "peak"),
    "asr_cer_vs_source_asr": ("asr", "cer_vs_source_asr"),
}


def stage_summary(root: Path, args: argparse.Namespace) -> None:
    """Per-condition min/mean/max over VC seeds of the headline metrics (machine-readable)."""
    data = read_json(root / "eval.json")
    identity, signal, asr = data["identity"]["per_item"], data["signal"], data["asr"] or {}

    def metric(name: str, source: str, key: str):
        if source == "identity":
            return identity.get(name, {}).get(key)
        if source == "asr":
            return asr.get(name, {}).get(key)
        block = {"f0": "f0_vs_source", "env": "envelope_vs_source", "wave": "waveform"}[source]
        return signal.get(name, {}).get(block, {}).get(key)

    groups: dict[str, list[str]] = {}
    for name, item in data["items"].items():
        if item["kind"] == "output" and item.get("role") != "repeatability":
            groups.setdefault(item["condition"], []).append(name)
        elif item["kind"] in ("old_output", "codec_roundtrip", "reference_original", "reference_matched",
                              "same_speaker", "anchor"):
            groups[name] = [name]
    summary = {}
    for group, names in groups.items():
        row = {"items": names, "role": data["items"][names[0]].get("role", data["items"][names[0]]["kind"])}
        for label, (source, key) in SUMMARY_METRICS.items():
            values = [v for v in (metric(n, source, key) for n in names) if v is not None]
            if values:
                row[label] = {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}                     if len(values) > 1 else values[0]
        summary[group] = row
    impostors = [identity[n]["cos_heldout_target"] for n, i in data["items"].items() if i["kind"] == "impostor"]
    calibration = {"impostor_count": len(impostors),
                   "label_caveat": "same_speaker/impostor are folder-label groups; human speaker identities were not independently verified",
                   "impostor_cos_heldout_target": {"min": min(impostors), "mean": sum(impostors) / len(impostors),
                                                   "max": max(impostors)} if impostors else None,
                   "same_speaker_cos_heldout_target": [identity[n]["cos_heldout_target"]
                                                       for n, i in data["items"].items() if i["kind"] == "same_speaker"],
                   "heldout_split_half_cos": identity["target_heldout"].get("split_half_cos"),
                   "source_singer_vs_target": identity["source_singer_other"].get("cos_heldout_target")}
    repeat = [r for r in read_json(root / "vc.json")["runs"] if r["role"] == "repeatability"]
    write_json(root / "summary.json", clean({"conditions": summary, "identity_calibration": calibration,
                                             "asr_tts_speech_vs_input_text": asr.get("tts_speech_vs_input_text"),
                                             "human_evaluation": read_json(root / "human_feedback.json")
                                             if (root / "human_feedback.json").exists() else {"status": "not_collected"},
                                             "repeatability_runs": [r["name"] for r in repeat]}))
    log(f"[summary] {len(summary)} groups")


# --------------------------------------------------------------------------- main


STAGES = {"setup": stage_setup, "clone": stage_clone, "cohort": stage_cohort, "refs": stage_refs, "vc": stage_vc,
          "roundtrip": stage_roundtrip, "eval": stage_eval, "summary": stage_summary, "pack": stage_pack}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma list of stages to run, in order.")
    parser.add_argument("--root", type=Path, help="Existing experiment folder to continue.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "singing")
    parser.add_argument("--target-audio", type=Path, required=True, help="Real target-speaker recording.")
    parser.add_argument("--source-audio", type=Path, required=True, help="Isolated source vocal.")
    parser.add_argument("--generic-ref", type=Path, required=True, help="Old caption-designed Irodori reference WAV.")
    parser.add_argument("--old-a-run", type=Path, required=True)
    parser.add_argument("--old-b-run", type=Path, required=True)
    parser.add_argument("--old-b-reference", type=Path, default=None,
                        help="Old B reference excerpt, to confirm it equals the new conditioning part sample for sample.")
    parser.add_argument("--wavlm-dir", type=Path, required=True)
    parser.add_argument("--whisper-dir", type=Path, default=None)
    parser.add_argument("--same-speaker", type=Path, nargs="*", default=[],
                        help="Other recordings of the target speaker (identity-proxy calibration only).")
    parser.add_argument("--cohort", type=Path, nargs="*", default=[],
                        help="Recordings of other speakers (impostors) for identity-proxy calibration.")
    parser.add_argument("--cohort-seconds", type=float, default=30.0)
    parser.add_argument("--ffmpeg", type=Path, default=PROJECT_ROOT / ".tools" / "ffmpeg" / "bin" / "ffmpeg.exe")
    parser.add_argument("--conditioning-seconds", type=float, default=25.0)
    parser.add_argument("--source-start", type=float, default=15.0)
    parser.add_argument("--source-duration", type=float, default=20.0)
    parser.add_argument("--clone-text", default=None, help="Default: the text of the old generic reference.")
    parser.add_argument("--tts-hf-checkpoint", default=DEFAULT_HF_CHECKPOINT)
    parser.add_argument("--tts-seed", type=int, default=0)
    parser.add_argument("--extra-clone-seeds", type=lambda s: [int(x) for x in s.split(",") if x], default=[],
                        help="Extra Irodori clone seeds, evaluated before SVC only.")
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1])
    parser.add_argument("--pack-seed", type=int, default=20261101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Re-run VC jobs even if outputs exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    for name in ("target_audio", "source_audio", "generic_ref", "old_a_run", "old_b_run", "old_b_reference",
                 "wavlm_dir", "whisper_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())  # subprocesses run with other working folders
    root = (args.root or args.output_dir / f"reference_comparison_{datetime.now():%Y%m%d-%H%M%S}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}")
    provenance_path = root / "provenance.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else {"stages": []}
    provenance.update({"irodori_revision": git_revision(PROJECT_ROOT), "seed_vc_revision": git_revision(SEEDVC_DIR),
                       "python": sys.version.split()[0]})
    for stage in stages:
        started = time.perf_counter()
        log(f"=== stage {stage} ===")
        STAGES[stage](root, args)
        provenance["stages"].append({"stage": stage, "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                     "seconds": round(time.perf_counter() - started, 1)})
        write_json(provenance_path, provenance)
    log(f"[done] {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
