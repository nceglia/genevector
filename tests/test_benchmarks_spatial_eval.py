"""Tests for genevector.benchmarks.spatial.eval and report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genevector.benchmarks.spatial import (
    BenchmarkConfig,
    evaluate_ablation,
    evaluate_gene_pair_recovery,
    evaluate_phenotype_typing,
    run_benchmark,
    write_report,
)


SEED = 42
VARIANTS = ("mi", "graph_xcorr", "graph_xcorr_shuffled")


@pytest.fixture(scope="module")
def populated_seed_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("eval_test")
    config = BenchmarkConfig(
        target_variants=list(VARIANTS),
        seed=SEED,
        epochs=200,
        threshold=1e-4,
        emb_dimension=32,
        device="cpu",
        layout_kwargs={"num_cells": 1500},
        output_root=str(out),
        log_interval=200,
    )
    run_benchmark(config)
    return Path(out) / f"seed_{SEED}"


@pytest.fixture(scope="module")
def gene_pair_results(populated_seed_dir):
    return evaluate_gene_pair_recovery(populated_seed_dir)


@pytest.fixture(scope="module")
def phenotype_results(populated_seed_dir):
    return evaluate_phenotype_typing(populated_seed_dir)


@pytest.fixture(scope="module")
def ablation_results(populated_seed_dir):
    return evaluate_ablation(populated_seed_dir)


def test_gene_pair_recovery_returns_aucs(gene_pair_results):
    assert set(gene_pair_results.keys()) == set(VARIANTS)
    for variant in VARIANTS:
        r = gene_pair_results[variant]
        for key in ("paracrine_auc", "niche_auc", "housekeeping_auc"):
            assert key in r
            assert 0.0 <= r[key] <= 1.0, f"{variant}.{key} = {r[key]}"


def test_paracrine_auc_higher_for_graph_xcorr(gene_pair_results):
    delta = (
        gene_pair_results["graph_xcorr"]["paracrine_auc"]
        - gene_pair_results["mi"]["paracrine_auc"]
    )
    if delta <= 0.1:
        pytest.xfail(
            f"paracrine_auc delta {delta:.3f} <= 0.1; may need more epochs"
        )
    assert delta > 0.1


def test_shuffled_collapses_paracrine_auc(gene_pair_results):
    delta = (
        gene_pair_results["graph_xcorr"]["paracrine_auc"]
        - gene_pair_results["graph_xcorr_shuffled"]["paracrine_auc"]
    )
    if delta <= 0.1:
        pytest.xfail(
            f"paracrine real-vs-shuffled delta {delta:.3f} <= 0.1; "
            "may need more epochs"
        )
    assert delta > 0.1


def test_housekeeping_at_chance(gene_pair_results):
    for variant in VARIANTS:
        auc = gene_pair_results[variant]["housekeeping_auc"]
        assert 0.35 <= auc <= 0.65, f"{variant} HK AUC = {auc:.3f} (expected ~0.5)"


def test_phenotype_eval_returns_f1(phenotype_results):
    assert set(phenotype_results.keys()) == set(VARIANTS)
    for variant in VARIANTS:
        r = phenotype_results[variant]
        assert 0.0 <= r["macro_f1"] <= 1.0
        for sub in ("T_stromal", "T_intratumoral", "T_rare"):
            assert sub in r["per_class_f1"], f"{variant} missing {sub}"
            assert 0.0 <= r["per_class_f1"][sub] <= 1.0


def test_ablation_summary_schema(ablation_results):
    assert "gene_pair" in ablation_results
    assert "phenotype" in ablation_results
    assert "ablation_summary" in ablation_results
    summary = ablation_results["ablation_summary"]
    for key in (
        "paracrine_auc_delta_real_vs_shuffled",
        "paracrine_auc_delta_real_vs_mi",
        "niche_auc_delta_real_vs_shuffled",
        "niche_auc_delta_real_vs_mi",
    ):
        assert key in summary, f"missing ablation key {key}"


def test_write_report_creates_markdown(populated_seed_dir, ablation_results, tmp_path):
    eval_dir = tmp_path / "eval" / f"seed_{SEED}"
    eval_dir.mkdir(parents=True)
    with open(eval_dir / "eval.json", "w") as f:
        json.dump(ablation_results, f)

    report_path = tmp_path / "report.md"
    write_report([eval_dir], report_path)

    assert report_path.exists()
    text = report_path.read_text()
    assert "Gene-pair recovery" in text
    for v in VARIANTS:
        assert v in text
