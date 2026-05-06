#!/usr/bin/env python
"""CLI for the GeneVector spatial benchmark evaluation.

Reads artifacts from prior ``run_spatial_benchmark.py`` invocations and
produces per-seed eval JSONs, a consolidated ``all_eval.json``, and a
Markdown report aggregating across seeds.

Examples
--------
    python scripts/run_spatial_eval.py --seeds 42 --output-root benchmarks_artifacts/spatial

    python scripts/run_spatial_eval.py --seeds 42 7 11 \\
        --output-root benchmarks_artifacts/spatial \\
        --report-path benchmarks_artifacts/spatial/eval/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genevector.benchmarks.spatial import (
    evaluate_ablation,
    write_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument(
        "--output-root",
        type=str,
        default="benchmarks_artifacts/spatial",
        help="Root directory containing seed_{seed}/ subdirs from TASK 028.",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Where to write the Markdown report. Defaults to "
        "{output_root}/eval/report.md.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    eval_root = output_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    consolidated: dict[str, dict] = {}
    seed_dirs = []
    for seed in args.seeds:
        seed_src = output_root / f"seed_{seed}"
        if not seed_src.exists():
            raise FileNotFoundError(f"Missing artifacts dir: {seed_src}")
        result = evaluate_ablation(seed_src)

        per_seed_dir = eval_root / f"seed_{seed}"
        per_seed_dir.mkdir(parents=True, exist_ok=True)
        with open(per_seed_dir / "eval.json", "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        consolidated[str(seed)] = result
        seed_dirs.append(per_seed_dir)

    with open(eval_root / "all_eval.json", "w") as f:
        json.dump(consolidated, f, indent=2, sort_keys=True)

    report_path = Path(args.report_path) if args.report_path else eval_root / "report.md"
    write_report(seed_dirs, report_path)
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
