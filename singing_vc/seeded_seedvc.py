"""Experiment-only wrapper: seed every RNG, then run ``seedvc_runner.main`` in this process.

Upstream Seed-VC ``inference.py`` has no seed option and draws its diffusion
noise from the global torch generator, so seeding before the runner makes a
run repeatable up to GPU kernel nondeterminism (fp16, cuDNN autotuning). The
production runner and pipeline are unchanged; this is used only by the
reference-comparison experiment.

  python singing_vc/seeded_seedvc.py --seed 0 [--cudnn-deterministic] -- <seedvc_runner args>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def seed_everything(seed: int, cudnn_deterministic: bool) -> dict:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {"seed": seed, "seeded": ["random", "numpy", "torch", "torch.cuda"],
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "limits": "fp16 GPU kernels may still differ run to run; repeatability is measured, not assumed"}


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        raise SystemExit("usage: seeded_seedvc.py --seed N [--cudnn-deterministic] -- <seedvc_runner args>")
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def main(argv: list[str] | None = None) -> int:
    own, runner_argv = split_argv(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cudnn-deterministic", action="store_true")
    args = parser.parse_args(own)

    from singing_vc import seedvc_runner

    report_path = seedvc_runner.parse_args(runner_argv).report_json.resolve()  # before the runner's chdir
    seeding = seed_everything(args.seed, args.cudnn_deterministic)
    code = seedvc_runner.main(runner_argv)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["seeding"] = seeding
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[seeded-seedvc] error: {exc}", file=sys.stderr)
        raise SystemExit(2)
