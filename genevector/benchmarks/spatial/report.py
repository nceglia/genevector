"""Aggregate per-seed eval results into a Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .eval import evaluate_ablation


def _collect_eval(seed_dirs: Iterable[Path]) -> list[dict]:
    out = []
    for d in seed_dirs:
        d = Path(d)
        cached = d / "eval.json"
        if cached.exists():
            with open(cached) as f:
                out.append(json.load(f))
        else:
            out.append(evaluate_ablation(d))
    return out


def _aggregate(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt(mean: float, std: float, bold: bool) -> str:
    if np.isnan(mean):
        cell = "—"
    else:
        cell = f"{mean:.3f} ± {std:.3f}"
    return f"**{cell}**" if bold else cell


def _gene_pair_table(evals: list[dict]) -> str:
    variants = sorted({v for e in evals for v in e["gene_pair"].keys()})
    metrics = ("paracrine_auc", "niche_auc", "housekeeping_auc")
    titles = {
        "paracrine_auc": "Paracrine AUC",
        "niche_auc": "Niche AUC",
        "housekeeping_auc": "Housekeeping AUC",
    }

    rows = []
    rows.append("## Gene-pair recovery")
    rows.append("")
    rows.append("| Variant | " + " | ".join(titles[m] for m in metrics) + " |")
    rows.append("|---" * (1 + len(metrics)) + "|")

    means_by_variant: dict[str, dict[str, tuple[float, float]]] = {}
    for v in variants:
        means_by_variant[v] = {}
        for m in metrics:
            vals = [e["gene_pair"].get(v, {}).get(m) for e in evals]
            means_by_variant[v][m] = _aggregate(vals)

    best_per_metric = {}
    for m in metrics:
        candidates = [(means_by_variant[v][m][0], v) for v in variants]
        candidates = [(mean, v) for mean, v in candidates if not np.isnan(mean)]
        best_per_metric[m] = max(candidates)[1] if candidates else None

    for v in variants:
        cells = [v]
        for m in metrics:
            mean, std = means_by_variant[v][m]
            cells.append(_fmt(mean, std, bold=(best_per_metric.get(m) == v)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _phenotype_table(evals: list[dict]) -> str:
    variants = sorted({v for e in evals for v in e["phenotype"].keys()})

    classes_present: set[str] = set()
    for e in evals:
        for v in variants:
            classes_present |= set(e["phenotype"].get(v, {}).get("per_class_f1", {}).keys())
    focus_classes = [
        c for c in ("T_intratumoral", "T_stromal", "T_rare") if c in classes_present
    ]

    rows = []
    rows.append("## Phenotype F1")
    rows.append("")
    header = ["Variant", "Macro F1"] + [f"F1[{c}]" for c in focus_classes]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("|---" * len(header) + "|")

    metric_means: dict[str, dict[str, tuple[float, float]]] = {}
    metric_keys = ["macro_f1"] + focus_classes
    for v in variants:
        metric_means[v] = {}
        macro_vals = [e["phenotype"].get(v, {}).get("macro_f1") for e in evals]
        metric_means[v]["macro_f1"] = _aggregate(macro_vals)
        for c in focus_classes:
            vals = [
                e["phenotype"].get(v, {}).get("per_class_f1", {}).get(c)
                for e in evals
            ]
            metric_means[v][c] = _aggregate(vals)

    best_per_metric = {}
    for m in metric_keys:
        candidates = [(metric_means[v][m][0], v) for v in variants]
        candidates = [(mean, v) for mean, v in candidates if not np.isnan(mean)]
        best_per_metric[m] = max(candidates)[1] if candidates else None

    for v in variants:
        cells = [v]
        for m in metric_keys:
            mean, std = metric_means[v][m]
            cells.append(_fmt(mean, std, bold=(best_per_metric.get(m) == v)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _ablation_table(evals: list[dict]) -> str:
    keys = (
        "paracrine_auc_delta_real_vs_shuffled",
        "paracrine_auc_delta_real_vs_mi",
        "niche_auc_delta_real_vs_shuffled",
        "niche_auc_delta_real_vs_mi",
    )

    rows = []
    rows.append("## Ablation summary (graph_xcorr deltas)")
    rows.append("")
    rows.append("| Delta | Mean ± Std |")
    rows.append("|---|---|")
    for k in keys:
        vals = [e.get("ablation_summary", {}).get(k) for e in evals]
        mean, std = _aggregate(vals)
        rows.append(f"| `{k}` | {_fmt(mean, std, bold=False)} |")
    return "\n".join(rows)


def write_report(seed_dirs: Iterable[str | Path], output_path: str | Path) -> Path:
    """Aggregate eval results across seeds and write a Markdown report.

    Parameters
    ----------
    seed_dirs : iterable of path-like
        Per-seed directories (each must contain ``eval.json`` *or* the raw
        artifacts needed by :func:`evaluate_ablation`).
    output_path : path-like
        Where to write the rendered ``report.md``.

    Returns
    -------
    Path
        The written report path.
    """
    seed_dirs = [Path(d) for d in seed_dirs]
    output_path = Path(output_path)
    evals = _collect_eval(seed_dirs)

    n_seeds = len(evals)
    parts = [f"# Spatial benchmark report (n_seeds={n_seeds})", ""]
    parts.append(_gene_pair_table(evals))
    parts.append("")
    parts.append(_phenotype_table(evals))
    parts.append("")
    parts.append(_ablation_table(evals))
    parts.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))
    return output_path
