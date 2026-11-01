"""Bounded causal ablations for "the DACVAE round trip sounds noisier" (codec only, no TTS model).

Runs inside the Irodori ``.venv``. The codec is loaded once; every variant feeds the
same 48 kHz mono signal (the wrapper's own mono mix + torchaudio resample), decodes with
the deterministic posterior mean, removes any conditioning gain exactly, and is compared
with that signal at the same rate and level.

  run   codec-free resample/I-O control, baseline (old path, reproduced), +6 dB, peak-safe
        up to +12 dB, -16 LUFS (model-card level) with gain restored, plus diagnostics:
        repeat, raw model-card code path, TF32 off, one fixed-seed posterior sample
  pack  level-matched listening pack: original, baseline, one chosen candidate

  .venv\\Scripts\\python.exe scripts\\codec_noise_diagnosis.py run --source a.wav --heldout b.wav
  .venv\\Scripts\\python.exe scripts\\codec_noise_diagnosis.py pack --root outputs\\singing\\codec_noise_X --candidate gain_up12_cap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from singing_vc import codec_diag as cd  # noqa: E402
from singing_vc import eval_metrics as em  # noqa: E402

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
LUFS_TARGET = -16.0
PEAK_CAP = 0.95
PACK_CEILING_DB = -1.0
PRINCIPAL = ("baseline", "gain_up6", "gain_up12_cap", "lufs_m16_restored")


class PathError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def same_file(a: Path, b: Path) -> bool:
    if os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b)):
        return True
    try:
        return os.path.samefile(a, b)  # hardlinks / symlinks
    except OSError:
        return False


def check_output(path: Path, root: Path, inputs: list[Path]) -> None:
    """Refuse writes outside ``root``, over an existing file, or onto any input (incl. hardlinks)."""
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise PathError(f"refusing to write outside the run root: {path}")
    if path.exists():
        raise PathError(f"refusing to overwrite existing file: {path}")
    for source in inputs:
        if same_file(path, source):
            raise PathError(f"output would overwrite input {source}")


def check_root(root: Path, inputs: list[Path]) -> None:
    resolved = root.resolve()
    for source in inputs:
        src = source.resolve()
        if resolved == src or resolved in src.parents:
            raise PathError(f"run root {root} must not contain or be the input {source}")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Runner:
    def __init__(self, device: str):
        from irodori_tts.local_env import configure_project_environment

        configure_project_environment()
        import torch
        import torchaudio

        from irodori_tts.codec import DACVAECodec

        self.torch, self.torchaudio = torch, torchaudio
        self.codec = DACVAECodec.load(repo_id=CODEC_REPO, device=device, normalize_db=None)
        self.rate = self.codec.sample_rate

    def to_codec_rate(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """Exactly the wrapper's preprocessing: channel mean, then torchaudio resample."""
        t = self.torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32)).unsqueeze(0)
        t = t.mean(dim=1, keepdim=True)
        if rate != self.rate:
            t = self.torchaudio.functional.resample(t, rate, self.rate)
        return t[0, 0].numpy()

    def resample(self, mono: np.ndarray, src: int, dst: int) -> np.ndarray:
        t = self.torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))[None]
        return self.torchaudio.functional.resample(t, src, dst)[0].numpy()

    def roundtrip(self, waveform, rate: int, *, sample_seed: int | None = None):
        """Wrapper encode (normalize off, ensure_max) + decode; returns (audio, latent)."""
        torch = self.torch
        wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        if wav.ndim == 1:
            wav = wav[None]
        if sample_seed is None:
            latent = self.codec.encode_waveform(wav, rate, normalize_db=None, ensure_max=True)
        else:  # diagnostic only: draw from the posterior instead of its mean
            with torch.inference_mode():
                x = wav[None].to(self.codec.device)
                z = self.codec.model.encoder(self.codec.model._pad(x))
                mean, scale = self.codec.model.quantizer.in_proj(z).chunk(2, dim=1)
                generator = torch.Generator(device=x.device).manual_seed(sample_seed)
                stdev = torch.nn.functional.softplus(scale) + 1e-4
                noise = torch.randn(mean.shape, generator=generator, device=x.device, dtype=mean.dtype)
                latent = (mean + noise * stdev).transpose(1, 2).contiguous()
        audio = self.codec.decode_latent(latent)[0, 0].float().cpu().numpy()
        return audio, latent.float().cpu().numpy()

    def raw_model_card(self, mono48: np.ndarray):
        """The model card's code path, verbatim apart from device placement."""
        torch, model = self.torch, self.codec.model
        with torch.inference_mode():
            x = torch.from_numpy(mono48)[None, None].to(self.codec.device)
            z = model.encoder(model._pad(x))
            z, _ = model.quantizer.in_proj(z).chunk(2, dim=1)
            y = model.decode(z)[0]
        return y[0].float().cpu().numpy(), z.transpose(1, 2).float().cpu().numpy()

    def lufs_gain_db(self, mono48: np.ndarray) -> float:
        """Gain the wrapper's audiotools -16 normalisation (+ ensure_max) applies; it is a pure scalar."""
        out = self.codec._normalize_loudness(self.torch.from_numpy(mono48), self.rate, LUFS_TARGET).numpy()
        return cd.scalar_gain_db(mono48, out)


def stage_run(args: argparse.Namespace) -> Path:
    inputs = [args.source.resolve(), args.heldout.resolve()]
    for path in inputs:
        if not path.is_file():
            raise PathError(f"missing input {path}")
    root = args.root or PROJECT_ROOT / "outputs" / "singing" / f"codec_noise_{datetime.now():%Y%m%d-%H%M%S}"
    check_root(root, inputs)
    root.mkdir(parents=True, exist_ok=False)  # fresh root only; nothing is ever deleted
    (root / "audio").mkdir()
    runner = Runner(args.device)
    torch, rate = runner.torch, runner.rate

    def save(name: str, audio: np.ndarray) -> Path:
        path = root / "audio" / f"{name}.wav"
        check_output(path, root, inputs)
        sf.write(str(path), np.asarray(audio, dtype=np.float32), rate, subtype="FLOAT")
        return path

    items = {"singing20s": args.source, "heldout_speech": args.heldout}
    report = {
        "command": sys.argv, "git_head": git_head(), "codec_repo": CODEC_REPO,
        "codec_weights": {"path": None, "sha256": None}, "codec_sample_rate": rate,
        "settings": {"encode": "wrapper encode_waveform(normalize_db=None, ensure_max=True), posterior mean",
                     "decode": "FP32 decode_latent, DACVAE watermark branch bypassed (alpha=0, model card)",
                     "lufs_target": LUFS_TARGET, "peak_cap": PEAK_CAP, "sample_seed": args.sample_seed,
                     "cudnn_tf32_default": bool(torch.backends.cudnn.allow_tf32), "torch": torch.__version__},
        "inputs": {}, "controls": {}, "variants": {}, "diagnostics": {}, "synthetic": {},
    }
    try:
        from huggingface_hub import hf_hub_download

        weights = Path(hf_hub_download(repo_id=CODEC_REPO, filename="weights.pth", local_files_only=True))
        report["codec_weights"] = {"path": str(weights), "snapshot": weights.parent.name, "sha256": sha256(weights)}
    except Exception as exc:  # recorded, not fatal
        report["codec_weights"]["error"] = repr(exc)

    for key, path in items.items():
        audio, in_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono48 = runner.to_codec_rate(audio, in_rate)
        report["inputs"][key] = {"path": str(path), "sha256": sha256(path), "sample_rate": in_rate,
                                 "channels": audio.shape[1], "seconds": audio.shape[0] / in_rate,
                                 "mono48_rms_dbfs": em.db(float(np.sqrt(np.mean(mono48.astype(np.float64) ** 2)))),
                                 "mono48_peak": float(np.max(np.abs(mono48))),
                                 "mono48_active_rms_db": em.active_rms_db(mono48, rate)}
        save(f"{key}__original48", mono48)

        # Codec-free control: write/read float WAV, 48k -> input rate -> 48k (the old pack resampled to 44.1k).
        tmp = save(f"{key}__io_control_float", mono48)
        back, _ = sf.read(str(tmp), dtype="float32")
        control = runner.resample(runner.resample(back, rate, in_rate), in_rate, rate)
        save(f"{key}__resample_control", control)
        report["controls"][key] = {"float_wav_io_max_abs_diff": float(np.max(np.abs(back - mono48))),
                                   "resample_roundtrip": cd.compare(mono48, control, rate)}

        variants = {}
        old, old_latent = runner.roundtrip(audio.T, in_rate)  # exactly the old call
        variants["baseline"] = {"gain_db": 0.0, "audio": old[: len(mono48)], "latent": old_latent}
        rep, rep_latent = runner.roundtrip(mono48, rate)
        report["diagnostics"].setdefault(key, {})["baseline_repeat_prepared48"] = {
            "max_abs_audio_diff": float(np.max(np.abs(rep[: len(mono48)] - variants["baseline"]["audio"]))),
            "max_abs_latent_diff": float(np.max(np.abs(rep_latent - old_latent)))}
        plans = {"gain_up6": cd.peak_safe_gain_db(mono48, 6.0, PEAK_CAP),
                 "gain_up12_cap": cd.peak_safe_gain_db(mono48, 12.0, PEAK_CAP),
                 "lufs_m16_restored": {"gain_db": runner.lufs_gain_db(mono48), "wanted": "audiotools -16 LUFS",
                                       "peak_cap": "1.0 (audiotools ensure_max), not the 0.95 used for manual gain"}}
        plans["lufs_m16_restored"]["conditioned_peak"] = float(
            np.max(np.abs(mono48)) * 10 ** (plans["lufs_m16_restored"]["gain_db"] / 20))
        for name, plan in plans.items():
            g = plan["gain_db"]
            dec, latent = runner.roundtrip(mono48 * np.float32(10 ** (g / 20)), rate)
            variants[name] = {"gain_db": g, "plan": plan, "audio": cd.restore_gain(dec[: len(mono48)], g),
                              "latent": latent, "decoded_peak_before_restore": float(np.max(np.abs(dec)))}
        g16 = plans["lufs_m16_restored"]["gain_db"]
        x16 = (mono48 * np.float32(10 ** (g16 / 20))).astype(np.float32)
        raw, raw_latent = runner.raw_model_card(x16)
        diag = report["diagnostics"][key]
        diag["raw_model_card_vs_wrapper_m16"] = {
            "max_abs_latent_diff": float(np.max(np.abs(raw_latent - variants["lufs_m16_restored"]["latent"]))),
            "max_abs_audio_diff_restored": float(np.max(np.abs(
                cd.restore_gain(raw[: len(mono48)], g16) - variants["lufs_m16_restored"]["audio"])))}
        before = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = False
            strict, _ = runner.roundtrip(audio.T, in_rate)
        finally:
            torch.backends.cudnn.allow_tf32 = before
        diag["baseline_tf32_off"] = {
            "tf32_before": before, "tf32_strict": False, "tf32_after": torch.backends.cudnn.allow_tf32,
            "max_abs_audio_diff": float(np.max(np.abs(strict[: len(mono48)] - variants["baseline"]["audio"]))),
            "diff_rms_dbfs": em.db(float(np.sqrt(np.mean((strict[: len(mono48)] - variants["baseline"]["audio"]) ** 2))))}
        sampled, _ = runner.roundtrip(x16, rate, sample_seed=args.sample_seed)
        sampled = cd.restore_gain(sampled[: len(mono48)], g16)
        diag["posterior_sample_m16_restored"] = cd.compare(mono48, sampled, rate)
        diag["posterior_sample_vs_mean_diff_rms_dbfs"] = em.db(float(np.sqrt(np.mean(
            (sampled.astype(np.float64) - variants["lufs_m16_restored"]["audio"]) ** 2))))
        save(f"{key}__posterior_sample_m16_restored", sampled)

        report["variants"][key] = {}
        for name, v in variants.items():
            save(f"{key}__{name}", v["audio"])
            row = {k: v[k] for k in ("gain_db", "plan", "decoded_peak_before_restore") if k in v}
            row["latent_shape"] = list(v["latent"].shape)
            row["latent_std"] = float(v["latent"].std())
            row["metrics"] = cd.compare(mono48, v["audio"], rate)
            report["variants"][key][name] = row
        print(f"[codec-noise] {key}: " + ", ".join(
            f"{n} active LSD {report['variants'][key][n]['metrics']['active']['log_spectral_distance_db']:.2f}dB"
            for n in variants), flush=True)

    # Out-of-domain probes: generated energy in silence, gap floors after a loud vs quiet tone.
    for name, probe in cd.synthetic_controls(rate).items():
        rows = {}
        for label, wanted in (("baseline", 0.0), ("gain_up12_cap", 12.0)):
            g = cd.peak_safe_gain_db(probe, wanted, PEAK_CAP)["gain_db"] if probe.any() else 0.0
            dec, _ = runner.roundtrip(probe * np.float32(10 ** (g / 20)), rate)
            dec = cd.restore_gain(dec[: len(probe)], g)
            sec, d64 = rate, dec.astype(np.float64)
            # Middle half of each 1 s segment, so tone onsets/offsets and their ringing are excluded.
            rows[label] = {"gain_db": g, "output_rms_dbfs": em.db(float(np.sqrt(np.mean(d64 ** 2)))),
                           "dc_offset": float(d64.mean()),
                           "per_second_mid_rms_dbfs": [
                               em.db(float(np.sqrt(np.mean(d64[i + sec // 4:i + 3 * sec // 4] ** 2))))
                               for i in range(0, len(d64) - sec + 1, sec)]}
        report["synthetic"][name] = {"note": "out-of-domain probe, not singing quality", **rows}

    write_json(root / "codec_noise.json", report)
    print(f"[codec-noise] wrote {root / 'codec_noise.json'}", flush=True)
    return root


def stage_pack(args: argparse.Namespace) -> None:
    """Named (not blind) level-matched pack: attenuation only, common active RMS, -1 dBFS peak ceiling."""
    root = args.root
    report = json.loads((root / "codec_noise.json").read_text(encoding="utf-8"))
    pack = root / "listening_pack"
    if pack.exists():
        raise PathError(f"listening pack already exists: {pack}; keep it and any ratings")
    if args.candidate not in PRINCIPAL[1:]:
        raise SystemExit(f"--candidate must be one of {PRINCIPAL[1:]}")
    inputs = [Path(v["path"]) for v in report["inputs"].values()]
    rate = report["codec_sample_rate"]
    pack.mkdir()
    manifest = {"blind": False, "sample_rate": rate, "subtype": "PCM_24", "peak_ceiling_dbfs": PACK_CEILING_DB,
                "candidate": args.candidate, "groups": {}}
    for key in report["inputs"]:  # each item is level matched within its own group
        names = {f"{key}_1_original": f"{key}__original48", f"{key}_2_baseline_old_path": f"{key}__baseline",
                 f"{key}_3_candidate_{args.candidate}": f"{key}__{args.candidate}"}
        audio = {n: sf.read(str(root / "audio" / f"{f}.wav"), dtype="float32")[0] for n, f in names.items()}
        levels = [em.active_rms_db(a, rate) for a in audio.values()]
        peaks = [float(np.max(np.abs(a))) for a in audio.values()]
        common, gains = em.attenuation_only_gains(levels, peaks, PACK_CEILING_DB)
        group = manifest["groups"][key] = {"common_active_rms_db": common, "files": {}}
        for (name, a), level, gain in zip(audio.items(), levels, gains):
            path = pack / f"{name}.wav"
            check_output(path, root, inputs)
            sf.write(str(path), (a * 10 ** (gain / 20)).astype(np.float32), rate, subtype="PCM_24")
            group["files"][path.name] = {"source": f"audio/{names[name]}.wav", "active_rms_db": level, "gain_db": gain}
        print(f"[codec-noise] pack {key} at {common:.1f} dB active RMS", flush=True)
    write_json(pack / "pack.json", manifest)


def stage_f0(args: argparse.Namespace) -> None:
    """Short pYIN check of the pack items against the original (index-aligned)."""
    from singing_vc import audio_tool

    root = args.root
    out = root / "f0.json"
    check_output(out, root, [])
    src = audio_tool.f0_contour(root / "audio" / "singing20s__original48.wav")
    result = {n: audio_tool.compare_contours(src, audio_tool.f0_contour(root / "audio" / f"singing20s__{n}.wav"), 0.0)
              for n in ("baseline", args.candidate)}
    write_json(out, result)
    print(json.dumps(clean(result), indent=1), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source", type=Path, required=True, help="sung excerpt (read-only)")
    run.add_argument("--heldout", type=Path, required=True, help="target speech recording (read-only)")
    run.add_argument("--root", type=Path, help="fresh output folder (default outputs/singing/codec_noise_<ts>)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--sample-seed", type=int, default=0)
    for name in ("pack", "f0"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--candidate", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    {"run": stage_run, "pack": stage_pack, "f0": stage_f0}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
