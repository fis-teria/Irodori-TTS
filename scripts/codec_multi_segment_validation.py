"""Held-out interval check of the model-card DACVAE round trip on other parts of the same song.

Codec only (no TTS, no Seed-VC, no training). Stages, each writing into one fresh root:

  select  fix intervals BEFORE inference: exact sample boundaries, source hash, activity facts,
          pre-registered silence rule + deterministic alternative, code/git provenance;
          extracts read-only excerpts into <root>/inputs
  run     one codec load; per interval: source mono48, baseline (old raw path), candidate
          (-16 LUFS + peak guard, posterior mean, inverse gain), codec-free resample control;
          corrected codec_diag metrics, crop/padding, DC/AC on source-defined quiet spans
  pack    48 kHz PCM24 attenuation-only active-RMS-matched triples: original anchor + A/B
          (seeded random method labels); key in <root>/listening_key/key.json
  audit-pack  re-measure the exported pack on one source-defined frame mask (read-only)
  f0      pYIN agreement with the source for baseline / candidate (pitch-tracking proxy only):
          common-voiced statistics plus source-voiced-denominator coverage; contours cached (.npz)
  verify-weights  reload from the pinned snapshot path and compare with the saved run audio

  .venv\\Scripts\\python.exe scripts\\codec_multi_segment_validation.py select --source song.wav
  .venv\\Scripts\\python.exe scripts\\codec_multi_segment_validation.py run --root <root>
  .venv\\Scripts\\python.exe scripts\\codec_multi_segment_validation.py pack --root <root>
  .venv\\Scripts\\python.exe scripts\\codec_multi_segment_validation.py f0 --root <root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import codec_noise_diagnosis as cnd  # noqa: E402
from singing_vc import codec_diag as cd  # noqa: E402
from singing_vc import eval_metrics as em  # noqa: E402
from singing_vc import multiseg as ms  # noqa: E402

JST = timezone(timedelta(hours=9))
DEFAULT_INTERVALS = ((55.0, 75.0), (100.0, 120.0), (145.0, 165.0))
PREVIOUS_INTERVAL = (15.0, 35.0)  # rated on 2026-11-01; excluded from held-out selection
EXPECTED_SNAPSHOT = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
EXPECTED_WEIGHTS_SHA = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"
PREVIOUS_ROOT = PROJECT_ROOT / "outputs" / "singing" / "codec_noise_20261101-185150"
PACK_SEED = 20261102
CODE_FILES = ("scripts/codec_multi_segment_validation.py", "scripts/codec_noise_diagnosis.py",
              "singing_vc/multiseg.py", "singing_vc/codec_diag.py", "singing_vc/eval_metrics.py",
              "singing_vc/audio_tool.py", "irodori_tts/codec.py")
METHODS = ("baseline", "candidate")


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_status_porcelain": git("status", "--porcelain"),
            "code_sha256": {f: cnd.sha256(PROJECT_ROOT / f) for f in CODE_FILES if (PROJECT_ROOT / f).is_file()},
            "python": sys.version.split()[0], "command": sys.argv,
            "time_jst": datetime.now(JST).isoformat(timespec="seconds")}


def previous_pack_hashes() -> dict:
    pack = PREVIOUS_ROOT / "listening_pack"
    return {p.name: cnd.sha256(p) for p in sorted(pack.iterdir()) if p.is_file()} if pack.is_dir() else {}


def segment_id(start: float, end: float) -> str:
    return f"s{int(start):03d}_{int(end):03d}"


def pinned_weights() -> tuple[Path, str]:
    """Cached weights at the pinned snapshot (no network), hash-checked before the codec is built."""
    from irodori_tts.local_env import configure_project_environment

    configure_project_environment()
    from huggingface_hub import hf_hub_download

    weights = Path(hf_hub_download(repo_id=cnd.CODEC_REPO, filename="weights.pth", revision=EXPECTED_SNAPSHOT,
                                   local_files_only=True))
    digest = cnd.sha256(weights)
    if weights.parent.name != EXPECTED_SNAPSHOT or digest != EXPECTED_WEIGHTS_SHA:
        raise SystemExit(f"codec checkpoint mismatch: {weights.parent.name} {digest}")
    return weights, digest


def stage_select(args: argparse.Namespace) -> Path:
    source = args.source.resolve()
    info = sf.info(str(source))
    rate, total = info.samplerate, info.frames
    duration = total / rate
    root = args.root or PROJECT_ROOT / "outputs" / "singing" / f"codec_multiseg_{datetime.now(JST):%Y%m%d-%H%M%S}"
    cnd.check_root(root, [source])
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    source_hash = cnd.sha256(source)

    def read(first: int, last: int) -> np.ndarray:
        audio, _ = sf.read(str(source), start=first, frames=last - first, always_2d=True, dtype="float32")
        return audio

    prev = ms.interval_samples(*PREVIOUS_INTERVAL, rate, total)
    taken_s, plan_rows = [PREVIOUS_INTERVAL], []
    for start, end in args.intervals:
        first, last = ms.interval_samples(start, end, rate, total)
        if ms.overlaps((first, last), prev) or any(ms.overlaps((first, last), (r["first_sample"], r["end_sample_exclusive"]))
                                                   for r in plan_rows):
            raise SystemExit(f"interval {start}-{end}s overlaps the rated interval or another held-out interval")
        facts = ms.activity(read(first, last), rate)
        row = {"requested_seconds": [start, end], "facts_requested": facts, "replaced": False}
        if facts["essentially_silent"]:  # deterministic replacement from source facts only
            length = end - start

            def active(s, length=length):
                a, b = ms.interval_samples(s, s + length, rate, total)
                return not ms.activity(read(a, b), rate)["essentially_silent"]
            alt = ms.alternative_start(start, length, duration, taken_s + [tuple(r["seconds"]) for r in plan_rows]
                                       + [tuple(x) for x in args.intervals if tuple(x) != (start, end)], active)
            if alt is None:
                raise SystemExit(f"interval {start}-{end}s is essentially silent and no alternative fits")
            start, end = alt, alt + length
            first, last = ms.interval_samples(start, end, rate, total)
            facts = ms.activity(read(first, last), rate)
            row["replaced"] = True
        audio = read(first, last)
        seg = segment_id(start, end)
        path = root / "inputs" / f"{seg}.wav"
        cnd.check_output(path, root, [source])
        subtype = info.subtype if info.subtype in ("PCM_16", "PCM_24") else "FLOAT"
        sf.write(str(path), audio, rate, subtype=subtype)
        back, _ = sf.read(str(path), always_2d=True, dtype="float32")
        if not np.array_equal(back, audio):
            raise RuntimeError(f"excerpt {path} does not read back bit-exact")
        row.update({"segment": seg, "seconds": [start, end], "first_sample": first, "end_sample_exclusive": last,
                    "samples": last - first, "facts": facts, "excerpt": f"inputs/{seg}.wav",
                    "excerpt_subtype": subtype, "excerpt_sha256": cnd.sha256(path)})
        plan_rows.append(row)
        print(f"[multiseg] {seg}: samples {first}-{last} rms {facts['rms_dbfs']:.1f} dBFS, "
              f">-50dBFS frames {facts['fraction_above_minus50dbfs']:.2f}, silent={facts['essentially_silent']}",
              flush=True)

    plan = {
        "created_before_codec_inference": True,
        "source": {"path": str(source), "sha256": source_hash, "sample_rate": rate, "channels": info.channels,
                   "frames": total, "seconds": duration, "subtype": info.subtype,
                   "role": "original singer vocal stem (not a target-voice conversion)"},
        "previous_rated_interval": {"seconds": list(PREVIOUS_INTERVAL), "first_sample": prev[0],
                                    "end_sample_exclusive": prev[1], "root": str(PREVIOUS_ROOT)},
        "previous_pack_sha256": previous_pack_hashes(),
        "selection_rule": {
            "fixed_intervals_seconds": [list(x) for x in args.intervals],
            "essentially_silent_if": f"mono RMS < {ms.SILENT_RMS_DBFS} dBFS or fraction of 25 ms frames above "
                                     f"{ms.ACTIVE_ABS_FLOOR_DBFS} dBFS < {ms.SILENT_ACTIVE_FRACTION}",
            "alternative": f"first {ms.ALTERNATIVE_STEP_SECONDS:g} s-grid start >= the silent start (then wrapping) "
                           "that fits, overlaps no rated/held-out interval and is not essentially silent",
            "labels": "intervals are named by time only; no chorus/verse/pitch labels are assumed"},
        "settings": {"codec_repo": cnd.CODEC_REPO, "expected_snapshot": EXPECTED_SNAPSHOT,
                     "expected_weights_sha256": EXPECTED_WEIGHTS_SHA,
                     "baseline": "old raw path: excerpt as read (stereo 44.1k) -> wrapper encode_waveform("
                                 "normalize_db=None, ensure_max=True) -> posterior mean -> decode",
                     "candidate": "wrapper mono mix + torchaudio resample to 48k -> audiotools -16 LUFS "
                                  "(ensure_max peak guard, pure scalar gain) -> posterior mean -> decode -> "
                                  "divide the same gain back out",
                     "control": "codec-free 48k -> source rate -> 48k torchaudio resample",
                     "pack_seed": PACK_SEED},
        "segments": plan_rows,
        "provenance": provenance(),
    }
    cnd.write_json(root / "selection_plan.json", plan)
    print(f"[multiseg] plan written: {root / 'selection_plan.json'}", flush=True)
    return root


def load_plan(root: Path) -> dict:
    return json.loads((root / "selection_plan.json").read_text(encoding="utf-8"))


def stage_run(args: argparse.Namespace) -> None:
    root = args.root
    plan = load_plan(root)
    source = Path(plan["source"]["path"])
    out_json = root / "multiseg_run.json"
    cnd.check_output(out_json, root, [source])
    if cnd.sha256(source) != plan["source"]["sha256"]:
        raise SystemExit("source hash changed since the selection plan")
    for row in plan["segments"]:
        if cnd.sha256(root / row["excerpt"]) != row["excerpt_sha256"]:
            raise SystemExit(f"excerpt {row['excerpt']} changed since the plan")
    weights, weights_sha = pinned_weights()  # verified before any model is constructed
    (root / "audio").mkdir(exist_ok=False)
    runner = cnd.Runner(args.device, codec_repo=str(weights))
    torch, rate = runner.torch, runner.rate
    inputs = [source] + [root / r["excerpt"] for r in plan["segments"]]

    def save(name: str, audio: np.ndarray) -> None:
        path = root / "audio" / f"{name}.wav"
        cnd.check_output(path, root, inputs)
        sf.write(str(path), np.asarray(audio, dtype=np.float32), rate, subtype="FLOAT")

    report = {"plan": "selection_plan.json", "codec_weights": {"path": str(weights), "snapshot": weights.parent.name,
                                                               "sha256": weights_sha,
                                                               "loaded_from": "this exact local path"},
              "codec_sample_rate": rate, "torch": torch.__version__,
              "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32), "device": args.device,
              "provenance": provenance(), "segments": {}}
    for row in plan["segments"]:
        seg = row["segment"]
        audio, in_rate = sf.read(str(root / row["excerpt"]), dtype="float32", always_2d=True)
        mono48 = runner.to_codec_rate(audio, in_rate)
        n = len(mono48)
        save(f"{seg}__original48", mono48)
        entry = {"input_samples": audio.shape[0], "input_rate": in_rate, "mono48_samples": n,
                 "mono48_peak": float(np.max(np.abs(mono48))),
                 "mono48_rms_dbfs": em.db(float(np.sqrt(np.mean(mono48.astype(np.float64) ** 2)))),
                 "mono48_active_rms_db": em.active_rms_db(mono48, rate), "methods": {}}

        old, latent = runner.roundtrip(audio.T, in_rate)  # exactly the old call
        old_c, crop = ms.crop_to_reference(old, n)
        entry["methods"]["baseline"] = {"gain_db": 0.0, "conditioned_peak": entry["mono48_peak"],
                                        "decoded_peak_before_restore": float(np.max(np.abs(old))),
                                        "latent_shape": list(latent.shape), "crop": crop, "audio": old_c}
        g = runner.lufs_gain_db(mono48)
        dec, latent = runner.roundtrip(mono48 * np.float32(10 ** (g / 20)), rate)
        dec_c, crop = ms.crop_to_reference(dec, n)
        entry["methods"]["candidate"] = {
            "gain_db": g, "gain_source": "audiotools -16 LUFS + ensure_max (peak <= 1.0)",
            "conditioned_peak": entry["mono48_peak"] * 10 ** (g / 20),
            "decoded_peak_before_restore": float(np.max(np.abs(dec))),
            "decoded_clipped_ratio_before_restore": float(np.mean(np.abs(dec) >= 0.999)),
            "latent_shape": list(latent.shape), "crop": crop, "audio": cd.restore_gain(dec_c, g)}
        control = runner.resample(runner.resample(mono48, rate, in_rate), in_rate, rate)
        control_c, crop = ms.crop_to_reference(control, n)
        entry["methods"]["resample_control"] = {"gain_db": 0.0, "crop": crop, "audio": control_c,
                                                "note": "codec-free: scale of resampling-only differences"}

        span = ms.longest_quiet_span(mono48.astype(np.float64), rate)
        dcac = {"definition": "source-defined: longest exact-zero run, else longest run of 10 ms frames "
                              f"< {ms.QUIET_SPAN_DBFS} dBFS; eroded {ms.QUIET_SPAN_ERODE_SECONDS}s per side, "
                              f">= {ms.QUIET_SPAN_MIN_SECONDS}s", "span": None, "stats": None}
        if span:
            a, b = span["start"], span["end"]
            dcac["span"] = {**span, "start_seconds_in_segment": a / rate, "end_seconds_in_segment": b / rate}
            dcac["stats"] = {"original48": cd.dc_ac_stats(mono48[a:b])}
        for name, m in entry["methods"].items():
            save(f"{seg}__{name}", m["audio"])
            m["metrics"] = cd.compare(mono48, m["audio"], rate)
            if span:
                dcac["stats"][name] = cd.dc_ac_stats(m["audio"][span["start"]:span["end"]])
        entry["dc_ac_quiet_span"] = dcac
        for m in entry["methods"].values():
            del m["audio"]
        report["segments"][seg] = entry
        print(f"[multiseg] {seg}: gain {g:+.2f} dB; active LSD baseline "
              f"{entry['methods']['baseline']['metrics']['active']['log_spectral_distance_db']:.2f} / candidate "
              f"{entry['methods']['candidate']['metrics']['active']['log_spectral_distance_db']:.2f} / control "
              f"{entry['methods']['resample_control']['metrics']['active']['log_spectral_distance_db']:.2f} dB",
              flush=True)
    report["post_checks"] = {"source_sha256_unchanged": cnd.sha256(source) == plan["source"]["sha256"],
                             "previous_pack_unchanged": previous_pack_hashes() == plan["previous_pack_sha256"],
                             "cudnn_tf32_after": bool(torch.backends.cudnn.allow_tf32)}
    cnd.write_json(out_json, report)
    print(f"[multiseg] wrote {out_json}; post checks {report['post_checks']}", flush=True)


def stage_pack(args: argparse.Namespace) -> None:
    root = args.root
    plan = load_plan(root)
    run = json.loads((root / "multiseg_run.json").read_text(encoding="utf-8"))
    pack, keydir = root / "listening_pack", root / "listening_key"
    if pack.exists() or keydir.exists():
        raise cnd.PathError("listening pack/key already exist; keep them and any ratings")
    source = Path(plan["source"]["path"])
    inputs = [source] + [root / r["excerpt"] for r in plan["segments"]]
    rate = run["codec_sample_rate"]
    segs = [r["segment"] for r in plan["segments"]]
    labels = ms.assign_ab(segs, PACK_SEED, METHODS)
    pack.mkdir()
    keydir.mkdir()
    public = {"sample_rate": rate, "subtype": "PCM_24", "peak_ceiling_dbfs": cnd.PACK_CEILING_DB,
              "level_matching": "attenuation only; common level over the source-defined active frames "
                                "(one identical frame mask for all three signals)",
              "blinding": {"blind": False, "method_labels_hidden": True, "original_identified": True,
                           "listeners_planned": 1, "note": "A/B hide which codec path is which; the original is "
                                                           "a named anchor, so this is not a fully blind test"},
              "groups": {}}
    key = {"seed": PACK_SEED, "assignment": labels, "groups": {}}
    for seg in segs:
        files = {f"{seg}_0_original": "original48", f"{seg}_A": labels[seg]["A"], f"{seg}_B": labels[seg]["B"]}
        audio = {n: sf.read(str(root / "audio" / f"{seg}__{src}.wav"), dtype="float32")[0] for n, src in files.items()}
        source48 = sf.read(str(root / "audio" / f"{seg}__original48.wav"), dtype="float32")[0]
        levels = ms.shared_mask_levels(source48, list(audio.values()), rate)["levels_db"]
        peaks = [float(np.max(np.abs(a))) for a in audio.values()]
        common, gains = em.attenuation_only_gains(levels, peaks, cnd.PACK_CEILING_DB)
        public["groups"][seg] = {"source_seconds": next(r["seconds"] for r in plan["segments"] if r["segment"] == seg),
                                 "files": {}}
        key["groups"][seg] = {"common_active_rms_db": common, "files": {}}
        for (name, a), level, gain in zip(audio.items(), levels, gains):
            if gain > 1e-9:
                raise RuntimeError("pack gains must be attenuation only")
            path = pack / f"{name}.wav"
            cnd.check_output(path, root, inputs)
            out = (a * 10 ** (gain / 20)).astype(np.float32)
            sf.write(str(path), out, rate, subtype="PCM_24")
            digest = cnd.sha256(path)
            public["groups"][seg]["files"][path.name] = {"sha256": digest}
            key["groups"][seg]["files"][path.name] = {"source": f"audio/{seg}__{files[name]}.wav",
                                                      "method": files[name], "active_rms_db": level,
                                                      "gain_db": gain, "peak_after_gain": float(np.max(np.abs(out))),
                                                      "sha256": digest}
    cnd.write_json(pack / "pack.json", public)
    cnd.write_json(keydir / "key.json", key)
    lines = ["試聴のお願い（コーデック往復・同じ曲の別区間）", "",
             "フォルダ内の3区間それぞれについて、次の3つを聴き比べてください。",
             "  <区間>_0_original.wav … 元の歌（基準）",
             "  <区間>_A.wav / <区間>_B.wav … どちらかが旧経路、どちらかが新しい候補（区間ごとに順序はランダム）",
             "", "音量は発声部分の平均が揃うよう減衰だけで調整しています。ヘッドホン推奨、音量は固定してください。",
             "対応表は別フォルダにあります。評価が終わるまで開かないでください。", "",
             "区間ごとに、気づいたことを自由に一言でかまいません（任意）:",
             "  ・A / B それぞれ、元の歌と比べてノイズ感・ざらつき・こもり/明るさ・子音・音程で気になる点",
             "  ・A と B でどちらが元に近いか（同じ / わからない も可）",
             "", "記入例は comments_template_ja.txt を使ってください。"]
    (pack / "README_試聴手順.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    template = []
    for seg in segs:
        template += [f"[{seg}]", "A の気になる点: ", "B の気になる点: ", "元に近いのは (A / B / 同じ / わからない): ", ""]
    (pack / "comments_template_ja.txt").write_text("\n".join(template), encoding="utf-8")
    print(f"[multiseg] pack written: {pack}", flush=True)


def stage_f0(args: argparse.Namespace) -> None:
    from singing_vc import audio_tool

    root = args.root
    out, cache = root / args.out, root / (Path(args.out).stem + "_contours.npz")
    cnd.check_output(out, root, [])
    cnd.check_output(cache, root, [])
    plan = load_plan(root)
    result = {"method": "librosa.pyin 16 kHz, hop 10 ms, 65-1100 Hz, index aligned; pitch-tracking agreement "
                        "only, not a noise or quality score",
              "coverage": "eval_metrics.f0_coverage: every source-voiced frame in the denominator; an unvoiced "
                          "or missing output frame counts as a failure",
              "contours": cache.name, "segments": {}}
    contours = {}
    for row in plan["segments"]:
        seg = row["segment"]
        src = contours[f"{seg}__original48"] = audio_tool.f0_contour(root / "audio" / f"{seg}__original48.wav")
        voiced = src[np.isfinite(src)]
        rowout = {"source_f0_hz_percentiles": {str(p): float(np.percentile(voiced, p)) for p in (5, 50, 95)}
                  if voiced.size else None}
        for name in METHODS:
            out_f0 = contours[f"{seg}__{name}"] = audio_tool.f0_contour(root / "audio" / f"{seg}__{name}.wav")
            rowout[name] = {"common_voiced": audio_tool.compare_contours(src, out_f0, 0.0),
                            "source_voiced_coverage": em.f0_coverage(src, out_f0)}
        result["segments"][seg] = rowout
    np.savez_compressed(cache, **contours)
    cnd.write_json(out, result)
    print(json.dumps(cnd.clean(result), indent=1), flush=True)


def stage_audit_pack(args: argparse.Namespace) -> None:
    """Re-measure the exported PCM24 pack on one source-defined frame mask per group (read-only)."""
    root = args.root
    out = root / "listening_pack_audit.json"
    cnd.check_output(out, root, [])
    key = json.loads((root / "listening_key" / "key.json").read_text(encoding="utf-8"))
    result = {"method": "frames active in audio/<seg>__original48.wav (em.frame_rms_db + em.active_mask); every "
                        "exported pack file measured on that identical frame set", "groups": {}}
    for seg, group in key["groups"].items():
        source48, rate = sf.read(str(root / "audio" / f"{seg}__original48.wav"), dtype="float32")
        names = list(group["files"])
        audio = [sf.read(str(root / "listening_pack" / n), dtype="float32")[0] for n in names]
        shared = ms.shared_mask_levels(source48, audio, rate)
        levels = dict(zip(names, shared["levels_db"]))
        result["groups"][seg] = {
            "mask_frames": shared["mask_frames"], "frames": shared["frames"], "levels_db": levels,
            "spread_db": max(levels.values()) - min(levels.values()),
            "sha256_matches_key": all(cnd.sha256(root / "listening_pack" / n) == group["files"][n]["sha256"]
                                      for n in names),
            "peak_dbfs": {n: em.db(float(np.max(np.abs(a)))) for n, a in zip(names, audio)}}
    cnd.write_json(out, result)
    print(json.dumps(cnd.clean(result), indent=1), flush=True)


def stage_verify_weights(args: argparse.Namespace) -> None:
    """Reload the codec from the pinned local snapshot path, redo baseline/candidate per segment and
    report the max difference to the saved run audio (provenance check; writes JSON only)."""
    root = args.root
    out = root / "pinned_reload_check.json"
    cnd.check_output(out, root, [])
    plan = load_plan(root)
    weights, digest = pinned_weights()
    runner = cnd.Runner(args.device, codec_repo=str(weights))
    rate = runner.rate
    result = {"weights": str(weights), "sha256": digest, "provenance": provenance(), "segments": {}}
    for row in plan["segments"]:
        seg = row["segment"]
        audio, in_rate = sf.read(str(root / row["excerpt"]), dtype="float32", always_2d=True)
        mono48 = runner.to_codec_rate(audio, in_rate)
        old, _ = runner.roundtrip(audio.T, in_rate)
        g = runner.lufs_gain_db(mono48)
        dec, _ = runner.roundtrip(mono48 * np.float32(10 ** (g / 20)), rate)
        fresh = {"baseline": old[: len(mono48)], "candidate": cd.restore_gain(dec[: len(mono48)], g)}
        result["segments"][seg] = {
            f"{name}_max_abs_diff_vs_saved": float(np.max(np.abs(
                a - sf.read(str(root / "audio" / f"{seg}__{name}.wav"), dtype="float32")[0])))
            for name, a in fresh.items()}
    cnd.write_json(out, result)
    print(json.dumps(cnd.clean(result), indent=1), flush=True)


def parse_interval(text: str) -> tuple[float, float]:
    a, b = text.split("-")
    return float(a), float(b)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--source", type=Path, required=True)
    sel.add_argument("--root", type=Path)
    sel.add_argument("--intervals", type=parse_interval, nargs="+", default=list(DEFAULT_INTERVALS),
                     help="start-end seconds, e.g. 55-75 100-120 145-165")
    for name in ("run", "verify-weights"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--device", default="cuda")
    for name in ("pack", "audit-pack"):
        sub.add_parser(name).add_argument("--root", type=Path, required=True)
    f0 = sub.add_parser("f0")
    f0.add_argument("--root", type=Path, required=True)
    f0.add_argument("--out", default="f0.json", help="new JSON name inside the root (never overwritten)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    {"select": stage_select, "run": stage_run, "pack": stage_pack, "audit-pack": stage_audit_pack,
     "f0": stage_f0, "verify-weights": stage_verify_weights}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
