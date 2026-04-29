"""Spatial benchmark evaluation.

Reads artifacts produced by :mod:`genevector.benchmarks.spatial.harness`
(``adata.h5ad``, ``ground_truth.json``, ``embedding_{variant}.vec``,
``meta_{variant}.json``) under a per-seed directory and answers the three
questions from the spatial benchmark plan:

1. **Gene-pair recovery** — ROC-AUC for paracrine, niche, and housekeeping
   pairs against random pairs in the gene embedding.
2. **Phenotype F1 under ambiguity** — macro-F1 + per-class F1 from
   :meth:`CellEmbedding.phenotype_probability`.
3. **Shuffled-graph ablation** — deltas between ``graph_xcorr``,
   ``graph_xcorr_shuffled``, and the ``mi`` baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from genevector.data import GeneVectorDataset
from genevector.embedding import CellEmbedding, GeneEmbedding


# ─── Internal helpers ──────────────────────────────────────────


def _list_variants(seed_dir: Path) -> list[str]:
    return sorted(p.stem.removeprefix("meta_") for p in seed_dir.glob("meta_*.json"))


def _load_ground_truth(seed_dir: Path) -> dict:
    with open(seed_dir / "ground_truth.json") as f:
        return json.load(f)


def _cosine(embed: GeneEmbedding, g1: str, g2: str) -> float | None:
    if g1 not in embed.embeddings or g2 not in embed.embeddings:
        return None
    v1 = np.asarray(embed.embeddings[g1]).reshape(1, -1)
    v2 = np.asarray(embed.embeddings[g2]).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0, 0])


def _pair_similarities(
    embed: GeneEmbedding, pairs: Iterable[tuple[str, str]]
) -> list[float]:
    sims = []
    for g1, g2 in pairs:
        s = _cosine(embed, g1, g2)
        if s is not None:
            sims.append(s)
    return sims


def _random_pair_similarities(
    embed: GeneEmbedding,
    n: int,
    exclude_pairs: set[tuple[str, str]],
    rng: np.random.Generator,
) -> list[float]:
    genes = list(embed.embeddings.keys())
    n_genes = len(genes)
    if n_genes < 2:
        return []
    sims: list[float] = []
    seen: set[tuple[str, str]] = set()
    max_attempts = n * 20
    attempts = 0
    while len(sims) < n and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, n_genes, size=2)
        if i == j:
            continue
        g1, g2 = genes[int(i)], genes[int(j)]
        key = (g1, g2)
        rev = (g2, g1)
        if key in exclude_pairs or rev in exclude_pairs:
            continue
        if key in seen or rev in seen:
            continue
        seen.add(key)
        sims.append(_cosine(embed, g1, g2))
    return sims


def _auc(true_sims: list[float], random_sims: list[float]) -> float:
    if not true_sims or not random_sims:
        return float("nan")
    y = np.concatenate([np.ones(len(true_sims)), np.zeros(len(random_sims))])
    s = np.concatenate([np.asarray(true_sims), np.asarray(random_sims)])
    return float(roc_auc_score(y, s))


def _discover_grafiti_markers(adata: ad.AnnData, gt: dict) -> dict[str, list[str]]:
    """Partition non-overlay markers among cell_types by max mean expression."""
    added = {g.upper() for g in gt.get("added_gene_names", [])}
    var_upper = [str(v).upper() for v in adata.var_names]
    grafiti_idx = [i for i, v in enumerate(var_upper) if v not in added]
    if not grafiti_idx:
        return {}

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    cell_types = sorted(set(adata.obs["cell_type"].astype(str)))

    type_mean = {}
    for ct in cell_types:
        mask = (adata.obs["cell_type"].astype(str) == ct).to_numpy()
        type_mean[ct] = (
            X[mask][:, grafiti_idx].mean(axis=0)
            if mask.any()
            else np.zeros(len(grafiti_idx))
        )

    out: dict[str, list[str]] = {ct: [] for ct in cell_types}
    for j, idx in enumerate(grafiti_idx):
        winner = max(cell_types, key=lambda ct: type_mean[ct][j])
        out[winner].append(var_upper[idx])
    return out


def _build_marker_dict(adata: ad.AnnData, gt: dict) -> dict[str, list[str]]:
    """Phenotype → markers map for ``phenotype_probability``."""
    type_markers = _discover_grafiti_markers(adata, gt)
    ligs = [p["ligand"].upper() for p in gt.get("paracrine_pairs", [])]
    recs = [p["receptor"].upper() for p in gt.get("paracrine_pairs", [])]
    niches = [n["gene"].upper() for n in gt.get("niche_genes", [])]
    trare = [g.upper() for g in gt.get("trare_genes", [])]

    return {
        "Tumor": type_markers.get("Tumor", []) + ligs,
        "Stromal": type_markers.get("Stromal", []),
        "B": type_markers.get("B cell", []),
        "Myeloid": type_markers.get("Myeloid", []),
        "T_stromal": type_markers.get("T cell", []) + recs,
        "T_intratumoral": type_markers.get("T cell", []) + recs + niches,
        "T_rare": type_markers.get("T cell", []) + recs + trare,
    }


_T_SUBTYPES = {"T_stromal", "T_intratumoral", "T_rare"}


def _coarsen_label(label: str) -> str:
    s = str(label)
    if s in _T_SUBTYPES:
        return s
    if s.startswith("Tumor"):
        return "Tumor"
    if s.startswith("Stromal"):
        return "Stromal"
    if s.startswith("B_"):
        return "B"
    if s.startswith("Myeloid"):
        return "Myeloid"
    return s


def _build_pairs_for_recovery(
    gt: dict,
) -> dict[str, list[tuple[str, str]]]:
    paracrine = [
        (p["ligand"].upper(), p["receptor"].upper())
        for p in gt.get("paracrine_pairs", [])
    ]
    ligs = [p["ligand"].upper() for p in gt.get("paracrine_pairs", [])]
    niches = [n["gene"].upper() for n in gt.get("niche_genes", [])]
    niche = [(n, l) for n in niches for l in ligs]
    hks = [g.upper() for g in gt.get("housekeeping_genes", [])]
    housekeeping = [(a, b) for i, a in enumerate(hks) for j, b in enumerate(hks) if i != j]
    return {"paracrine": paracrine, "niche": niche, "housekeeping": housekeeping}


# ─── Public API ────────────────────────────────────────────────


def evaluate_gene_pair_recovery(
    seed_dir: str | Path,
    n_random_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """Test 1: ROC-AUC of true spatial pairs vs random pairs in gene embedding.

    Parameters
    ----------
    seed_dir : path-like
        Per-seed directory produced by ``run_benchmark``.
    n_random_pairs : int
        Number of random gene pairs to score against.
    seed : int
        RNG seed for random-pair sampling.

    Returns
    -------
    dict
        ``{variant: {"paracrine_auc", "niche_auc", "housekeeping_auc",
        "n_paracrine", "n_niche", "n_housekeeping", "n_random"}}``.
    """
    seed_dir = Path(seed_dir)
    gt = _load_ground_truth(seed_dir)
    adata = ad.read_h5ad(seed_dir / "adata.h5ad")
    variants = _list_variants(seed_dir)

    pairs_by_class = _build_pairs_for_recovery(gt)
    exclude = {p for ps in pairs_by_class.values() for p in ps}

    results: dict[str, dict] = {}
    for variant in variants:
        embed_path = str(seed_dir / f"embedding_{variant}.vec")
        dataset = GeneVectorDataset(adata.copy(), target="mi", use_cache=False)
        embed = GeneEmbedding(embed_path, dataset, vector="average")

        rng = np.random.default_rng(seed)
        random_sims = _random_pair_similarities(embed, n_random_pairs, exclude, rng)

        para_sims = _pair_similarities(embed, pairs_by_class["paracrine"])
        niche_sims = _pair_similarities(embed, pairs_by_class["niche"])
        hk_sims = _pair_similarities(embed, pairs_by_class["housekeeping"])

        results[variant] = {
            "paracrine_auc": _auc(para_sims, random_sims),
            "niche_auc": _auc(niche_sims, random_sims),
            "housekeeping_auc": _auc(hk_sims, random_sims),
            "n_paracrine": len(para_sims),
            "n_niche": len(niche_sims),
            "n_housekeeping": len(hk_sims),
            "n_random": len(random_sims),
        }
    return results


def evaluate_phenotype_typing(
    seed_dir: str | Path,
    method: str = "normalized_exponential",
    temperature: float = 0.05,
) -> dict:
    """Test 2: macro-F1 and per-phenotype F1 from ``phenotype_probability``.

    Parameters
    ----------
    seed_dir : path-like
        Per-seed directory.
    method : str
        Probability conversion method passed through to
        ``phenotype_probability``.
    temperature : float
        Temperature for the normalized-exponential method.

    Returns
    -------
    dict
        ``{variant: {"macro_f1", "per_class_f1", "confusion_matrix",
        "labels"}}``.
    """
    seed_dir = Path(seed_dir)
    gt = _load_ground_truth(seed_dir)
    adata_disk = ad.read_h5ad(seed_dir / "adata.h5ad")
    variants = _list_variants(seed_dir)

    markers = _build_marker_dict(adata_disk, gt)

    results: dict[str, dict] = {}
    for variant in variants:
        adata = adata_disk.copy()
        dataset = GeneVectorDataset(adata, target="mi", use_cache=False)
        embed_path = str(seed_dir / f"embedding_{variant}.vec")
        gene_embed = GeneEmbedding(embed_path, dataset, vector="average")
        cell_embed = CellEmbedding(dataset, gene_embed)
        adata_gv = cell_embed.get_adata()
        adata_gv = cell_embed.phenotype_probability(
            adata_gv,
            markers,
            method=method,
            target_col="genevector",
            temperature=temperature,
        )

        true_raw = adata_gv.obs["phenotype"].astype(str).to_numpy()
        true_labels = np.array([_coarsen_label(x) for x in true_raw])
        pred_labels = adata_gv.obs["genevector"].astype(str).to_numpy()

        labels = sorted(set(true_labels) | set(pred_labels))
        macro = f1_score(
            true_labels, pred_labels, labels=labels,
            average="macro", zero_division=0,
        )
        per = f1_score(
            true_labels, pred_labels, labels=labels,
            average=None, zero_division=0,
        )
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)

        results[variant] = {
            "macro_f1": float(macro),
            "per_class_f1": {labels[i]: float(per[i]) for i in range(len(labels))},
            "confusion_matrix": cm.tolist(),
            "labels": list(labels),
        }
    return results


def evaluate_ablation(seed_dir: str | Path) -> dict:
    """Test 3: side-by-side comparison of ``graph_xcorr`` vs
    ``graph_xcorr_shuffled`` against the ``mi`` baseline.

    Returns
    -------
    dict
        ``{"gene_pair", "phenotype", "ablation_summary"}`` where
        ``ablation_summary`` exposes paracrine and niche AUC deltas.
    """
    gp = evaluate_gene_pair_recovery(seed_dir)
    ph = evaluate_phenotype_typing(seed_dir)

    summary: dict[str, float] = {}
    real = gp.get("graph_xcorr")
    shuf = gp.get("graph_xcorr_shuffled")
    base = gp.get("mi")
    if real and shuf:
        summary["paracrine_auc_delta_real_vs_shuffled"] = (
            real["paracrine_auc"] - shuf["paracrine_auc"]
        )
        summary["niche_auc_delta_real_vs_shuffled"] = (
            real["niche_auc"] - shuf["niche_auc"]
        )
    if real and base:
        summary["paracrine_auc_delta_real_vs_mi"] = (
            real["paracrine_auc"] - base["paracrine_auc"]
        )
        summary["niche_auc_delta_real_vs_mi"] = (
            real["niche_auc"] - base["niche_auc"]
        )

    return {"gene_pair": gp, "phenotype": ph, "ablation_summary": summary}
