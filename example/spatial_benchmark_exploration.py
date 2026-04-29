# %% Cell 0 — Header
"""
Interactive exploration of GeneVector spatial benchmark results.

Reads artifacts from a prior `python scripts/run_spatial_benchmark.py --seed 42` run
(default 3000 epochs, full 10k-cell FOV1, all 5 variants) under
`benchmarks_artifacts/spatial/seed_42/`. Walks through:

  1. Sanity-check the dataset — confirm phenotype labels, marker layout,
     ground-truth coverage.
  2. Inspect each gene embedding by variant.
  3. Reproduce the eval pipeline cell-by-cell so the failure modes are
     visible inline (paracrine AUC < 0.5; T_rare F1 = 0).
  4. Provide handles for tweaking — alternative pair selection, alternative
     marker dicts, alternative similarity scoring.

Run interactively: open in VS Code / Cursor and execute cells via `#%%`.
"""

ARTIFACTS_ROOT = "benchmarks_artifacts/spatial"
SEED = 42


# %% Cell 1 — Imports + paths
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

seed_dir = Path(ARTIFACTS_ROOT) / f"seed_{SEED}"
if not seed_dir.exists():
    raise FileNotFoundError(
        f"Missing artifacts dir: {seed_dir}\n"
        f"Run the benchmark first:\n"
        f"  python scripts/run_spatial_benchmark.py --seed {SEED}"
    )

print(f"Artifacts root: {seed_dir}")
print(f"Files present : {sorted(p.name for p in seed_dir.iterdir())}")


# %% [markdown]
# # Load all artifacts
#
# We loaded:
#   - `adata.h5ad` — the synthetic FOV with 23 diagnostic genes overlaid (LIG/REC/NICHE/TRARE/HK)
#     and T cells split into T_stromal/T_intratumoral/T_rare by local tumor density.
#   - `ground_truth.json` — the diagnostic schema (paracrine pairs, niche/trare/HK gene lists, T-subtype counts).
#   - `graph_real.npz`, `graph_shuffled.npz` — spatial connectivity (real k-NN at radius 1.5) and
#     a degree-preserving shuffle. The shuffled graph is the critical control: if `graph_xcorr` wins
#     persist under shuffling, the spatial structure isn't the source.
#   - `embedding_{variant}.vec` for each of {mi, pearson, spearman, graph_xcorr, graph_xcorr_shuffled}.
#   - `meta_{variant}.json` — final loss, wall time, epochs run per variant.

# %% Cell 2 — Load artifacts
def _read_vec_file(path: Path) -> dict[str, np.ndarray]:
    out = {}
    lines = path.read_text().splitlines()
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        gene = parts[0]
        out[gene] = np.array([float(x) for x in parts[1:]], dtype=np.float64)
    return out


def _load_vec(path: Path) -> dict[str, np.ndarray]:
    """Average input + output weights to match the eval's `vector='average'`."""
    primary = _read_vec_file(path)
    secondary_path = path.with_name(path.stem + "2.vec")
    if not secondary_path.exists():
        return primary
    secondary = _read_vec_file(secondary_path)
    common = primary.keys() & secondary.keys()
    return {g: 0.5 * (primary[g] + secondary[g]) for g in common}


adata = ad.read_h5ad(seed_dir / "adata.h5ad")
gt = json.loads((seed_dir / "ground_truth.json").read_text())
graph_real = sparse.load_npz(seed_dir / "graph_real.npz")
graph_shuffled = sparse.load_npz(seed_dir / "graph_shuffled.npz")

variants = sorted(p.stem.removeprefix("meta_") for p in seed_dir.glob("meta_*.json"))
embeddings_by_variant = {
    v: _load_vec(seed_dir / f"embedding_{v}.vec") for v in variants
}
meta_by_variant = {
    v: json.loads((seed_dir / f"meta_{v}.json").read_text()) for v in variants
}

print(f"Variants loaded : {variants}")
print(f"Adata shape     : {adata.shape}")
print(f"GT keys         : {sorted(gt.keys())}")
print(f"Graph real nnz  : {graph_real.nnz}, shuffled nnz: {graph_shuffled.nnz}")


# %% [markdown]
# # Sanity check: do the obs columns match what the eval expects?
#
# Two common bugs we want to catch immediately:
#
# **(1) Missing `phenotype_coarse`.** TASK 027's overlay should have created this column
#     to preserve the original `T` label after splitting into 3 subtypes. If it's missing,
#     the eval's coarsening logic is reading a column that doesn't exist.
#
# **(2) Phenotype label format mismatch.** The eval's `_discover_grafiti_markers` partitions
#     by `obs["cell_type"]`, but the overlay writes to `obs["phenotype"]`. If `cell_type` doesn't
#     exist or uses different label strings (e.g. "B" vs "B cell"), every grafiti-marker bucket
#     comes back empty and only the 23 diagnostic genes drive classification.
#
# Also confirm that T_stromal / T_intratumoral / T_rare are present and non-trivially populated.
# T_rare = 0 cells would explain the F1[T_rare] = 0 across all variants except the noisy shuffled.

# %% Cell 3 — Sanity-check phenotype labels and obs columns
print("obs columns:", adata.obs.columns.tolist())
print()
print("phenotype value_counts:")
print(adata.obs["phenotype"].value_counts())
print()
if "phenotype_coarse" in adata.obs.columns:
    print("phenotype_coarse value_counts:")
    print(adata.obs["phenotype_coarse"].value_counts())
else:
    print("MISMATCH: `phenotype_coarse` column is missing.")
print()
if "cell_type" in adata.obs.columns:
    print("cell_type value_counts:")
    print(adata.obs["cell_type"].value_counts())
    expected = {"Tumor", "T cell", "B cell", "Myeloid", "Stromal"}
    actual = set(adata.obs["cell_type"].astype(str))
    if expected.issubset(actual):
        print("# OK: all 5 expected cell_type labels present.")
    else:
        print(f"# MISMATCH: cell_type missing {expected - actual}, has extras {actual - expected}.")
else:
    print("# MISMATCH: `cell_type` column is missing — eval's marker discovery will fail.")

t_subs = ["T_stromal", "T_intratumoral", "T_rare"]
pheno_counts = adata.obs["phenotype"].value_counts()
print()
print("T-subtype counts:")
for t in t_subs:
    n = int(pheno_counts.get(t, 0))
    flag = "" if n > 5 else "  ← thin/empty"
    print(f"  {t:18s} {n:5d}{flag}")


# %% [markdown]
# # Are the diagnostic genes actually present in each embedding?
#
# If `LIG_0` is missing from `embedding_graph_xcorr.vec`, then every paracrine AUC is
# computed against `n_paracrine = 0` pairs, which is meaningless. Worth confirming for
# all 23 overlay genes × 5 variants before reading anything into the AUC numbers.

# %% Cell 4 — Confirm overlay genes are in the embedding
overlay_genes = list(gt["added_gene_names"])

coverage = pd.DataFrame(
    index=overlay_genes,
    columns=variants,
    data="✗",
)
for v in variants:
    keys = embeddings_by_variant[v].keys()
    for g in overlay_genes:
        if g.upper() in keys:
            coverage.loc[g, v] = "✓"

print(coverage.to_string())
print()

categories = {
    "LIG": [g for g in overlay_genes if g.startswith("LIG_")],
    "REC": [g for g in overlay_genes if g.startswith("REC_")],
    "NICHE": [g for g in overlay_genes if g.startswith("NICHE_")],
    "TRARE": [g for g in overlay_genes if g.startswith("TRARE_")],
    "HK": [g for g in overlay_genes if g.startswith("HK_")],
}
print("Coverage by category:")
for v in variants:
    line = [f"  {v:28s}"]
    for cat, genes in categories.items():
        n_present = sum(1 for g in genes if g.upper() in embeddings_by_variant[v])
        line.append(f"{cat}={n_present}/{len(genes)}")
    print(" ".join(line))


# %% [markdown]
# # Did each variant actually converge?
#
# From the meta JSONs:
#   - mi: final loss ~19 (3000 epochs)
#   - pearson: ~1187
#   - spearman: ~1.2
#   - graph_xcorr (real): ~2.3
#   - graph_xcorr_shuffled: ~0.10
#
# The shuffled run converging to ~0.10 is *expected behavior for the control* — when the graph is
# randomized, the cross-correlation targets average toward zero, so the model trivially fits them.
# This is what we want from a control: the spatial signal is destroyed.
#
# But it has a side effect: the shuffled run's gene embedding may collapse, and similarity scores
# in collapsed space are noisy. The shuffled control's wins on niche/F1 are likely artifacts of
# this collapse rather than evidence the shuffled graph "learned" something.

# %% Cell 5 — Loss curves and convergence
meta_df = pd.DataFrame(
    [
        {
            "variant": v,
            "epochs_run": m["epochs_run"],
            "final_loss": m["final_loss"],
            "wall_seconds": round(m["wall_time_seconds"], 2),
            "graph_variant": m["graph_variant"],
        }
        for v, m in meta_by_variant.items()
    ]
).set_index("variant")
print(meta_df.to_string())


# %% [markdown]
# # Detect embedding collapse via vector norms
#
# If a variant collapsed (all genes near origin, or all genes near a single point), the cosine
# similarity-based eval will be unreliable. Two diagnostics:
#   - distribution of `||v_i||` across genes (mean, std)
#   - mean pairwise cosine similarity (collapsed → near 1; healthy → centered around 0)

# %% Cell 6 — Vector norm distributions per variant
collapse_rows = []
for v in variants:
    vecs = np.stack(list(embeddings_by_variant[v].values()), axis=0)
    norms = np.linalg.norm(vecs, axis=1)
    cos = cosine_similarity(vecs)
    iu = np.triu_indices_from(cos, k=1)
    collapse_rows.append(
        {
            "variant": v,
            "n_genes": vecs.shape[0],
            "norm_mean": norms.mean(),
            "norm_std": norms.std(),
            "norm_min": norms.min(),
            "norm_max": norms.max(),
            "mean_pairwise_cos": cos[iu].mean(),
            "abs_mean_pairwise_cos": np.abs(cos[iu]).mean(),
        }
    )
collapse_df = pd.DataFrame(collapse_rows).set_index("variant")
print(collapse_df.round(4).to_string())

fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 2.5), sharey=True)
if len(variants) == 1:
    axes = [axes]
for ax, v in zip(axes, variants):
    norms = np.linalg.norm(
        np.stack(list(embeddings_by_variant[v].values()), axis=0), axis=1
    )
    ax.hist(norms, bins=30, color="steelblue", alpha=0.85)
    ax.set_title(v, fontsize=9)
    ax.set_xlabel("||v||")
fig.suptitle("Gene-vector norms per variant", fontsize=11)
fig.tight_layout()
plt.show()


# %% [markdown]
# # The puzzle: paracrine AUC is below chance for every variant.
#
# Ground truth says LIG_i ↔ REC_i should be highly similar in `graph_xcorr`'s embedding (they
# co-occur across spatial neighborhoods). Yet AUC was 0.097 — *worse than chance*. AUCs below 0.5
# usually mean the scoring direction is flipped: the true pairs are *less* similar than random
# pairs, which is the opposite of what we want.
#
# One hypothesis: LIG and REC are *never co-expressed in the same cell*, so within-cell methods
# (mi, pearson, spearman) place them in opposite regions of vector space (high anti-correlation).
# graph_xcorr is supposed to counteract this by pulling them together via neighborhood signal.
# If the training signal isn't strong enough, the within-cell anti-coupling dominates.
#
# Direct check below: compute the cosine similarity for each LIG_i / REC_i pair across all
# variants. Negative numbers near -1 confirm anti-correlation.

# %% Cell 7 — Paracrine pair similarities, by hand
def _pair_cos(vecs: dict[str, np.ndarray], a: str, b: str) -> float:
    if a not in vecs or b not in vecs:
        return float("nan")
    return float(
        cosine_similarity(vecs[a].reshape(1, -1), vecs[b].reshape(1, -1))[0, 0]
    )


paracrine_pairs = [
    (p["ligand"].upper(), p["receptor"].upper()) for p in gt["paracrine_pairs"]
]
para_rows = []
for lig, rec in paracrine_pairs:
    row = {"pair": f"{lig} ↔ {rec}"}
    for v in variants:
        row[v] = _pair_cos(embeddings_by_variant[v], lig, rec)
    para_rows.append(row)
para_df = pd.DataFrame(para_rows).set_index("pair")

rng = np.random.default_rng(0)
all_genes = sorted({g for d in embeddings_by_variant.values() for g in d})
n_random = 5
random_rows = []
for _ in range(n_random):
    a, b = rng.choice(all_genes, size=2, replace=False)
    row = {"pair": f"random: {a} ↔ {b}"}
    for v in variants:
        row[v] = _pair_cos(embeddings_by_variant[v], a, b)
    random_rows.append(row)

para_full = pd.concat([para_df, pd.DataFrame(random_rows).set_index("pair")])
print(para_full.round(3).to_string())

fig, ax = plt.subplots(figsize=(1.3 * len(variants) + 2, 0.4 * len(para_full) + 1))
sns.heatmap(
    para_full.astype(float),
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0.0,
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "cosine similarity"},
    ax=ax,
)
ax.set_title("Paracrine LIG ↔ REC similarity (top) vs random (bottom)")
plt.tight_layout()
plt.show()


# %% [markdown]
# # The niche result IS working — let's confirm.
#
# graph_xcorr niche AUC = 0.470 vs mi 0.038. That's a +0.43 absolute lift, and the shuffled
# control collapses it to 0.219. This is the part of the benchmark that's behaving as predicted:
# `graph_xcorr` picks up neighborhood-mediated coupling that within-cell MI cannot.
#
# We score NICHE_i against the LIG_i genes (Tumor markers by construction). Each NICHE_i has 5
# possible "target" Tumor markers, so 5 × 5 = 25 ground-truth pairs.

# %% Cell 8 — Niche-gene similarity to tumor markers
niches = [n["gene"].upper() for n in gt["niche_genes"]]
ligs = [p["ligand"].upper() for p in gt["paracrine_pairs"]]

niche_rows = []
for n in niches:
    for l in ligs:
        row = {"pair": f"{n} ↔ {l}"}
        for v in variants:
            row[v] = _pair_cos(embeddings_by_variant[v], n, l)
        niche_rows.append(row)
niche_df = pd.DataFrame(niche_rows).set_index("pair")
print(niche_df.round(3).to_string())

print()
print("Mean niche similarity by variant (true pairs only):")
print(niche_df.mean(axis=0).round(3).to_string())

fig, ax = plt.subplots(figsize=(1.3 * len(variants) + 2, 0.32 * len(niche_df) + 1))
sns.heatmap(
    niche_df.astype(float),
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0.0,
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "cosine similarity"},
    ax=ax,
)
ax.set_title("Niche ↔ Tumor-marker (LIG) similarity")
plt.tight_layout()
plt.show()


# %% [markdown]
# # Why is F1[T_rare] zero for every variant except the noisy shuffled?
#
# Three possibilities:
#   (a) T_rare cells are too few to score (look at counts in cell 3).
#   (b) The marker dict for T_rare doesn't actually distinguish them — TRARE_0..TRARE_2 may not
#       be making it through `_build_marker_dict` if the discovery heuristic fails (cell 3).
#   (c) `phenotype_probability` always assigns T_rare candidates to T_intratumoral or T_stromal
#       because the shared T markers dominate over the 3 distinguishing TRARE genes.
#
# Below we run `phenotype_probability` for one variant and inspect the per-cell probability
# matrix on cells whose true label is T_rare. If the probability for T_rare is consistently
# lower than for T_intratumoral, that's hypothesis (c).

# %% Cell 9 — Per-cell phenotype probability inspection
from genevector.benchmarks.spatial.eval import _build_marker_dict
from genevector.data import GeneVectorDataset
from genevector.embedding import CellEmbedding, GeneEmbedding

INSPECT_VARIANT = "graph_xcorr" if "graph_xcorr" in variants else variants[0]

markers = _build_marker_dict(adata, gt)
adata_inspect = adata.copy()
dataset = GeneVectorDataset(adata_inspect, target="mi", use_cache=False)
gene_embed = GeneEmbedding(
    str(seed_dir / f"embedding_{INSPECT_VARIANT}.vec"), dataset, vector="average"
)
cell_embed = CellEmbedding(dataset, gene_embed)
adata_gv = cell_embed.get_adata()
adata_gv = cell_embed.phenotype_probability(
    adata_gv,
    markers,
    method="normalized_exponential",
    target_col="genevector",
    temperature=0.05,
)

prob_cols = [c for c in adata_gv.obs.columns if c.endswith("Pseudo-probability")]
trare_mask = adata_gv.obs["phenotype"].astype(str) == "T_rare"
print(f"Inspecting variant: {INSPECT_VARIANT}")
print(f"True T_rare cells in adata_gv: {int(trare_mask.sum())}")

if trare_mask.any():
    rare_means = adata_gv.obs.loc[trare_mask, prob_cols].mean()
    rare_means.index = [c.replace(" Pseudo-probability", "") for c in rare_means.index]
    print()
    print("Mean phenotype probability across true T_rare cells:")
    print(rare_means.sort_values(ascending=False).round(4).to_string())

    print()
    print("Predicted label distribution for true T_rare cells:")
    print(adata_gv.obs.loc[trare_mask, "genevector"].value_counts().to_string())
else:
    print("No T_rare cells in get_adata() output — check filtering in CellEmbedding.")


# %% [markdown]
# # Inspect what `_build_marker_dict` actually produces.
#
# The eval relies on this function to map phenotype → list of marker genes for
# `phenotype_probability`. If the discovery heuristic in `_discover_grafiti_markers` fails
# (because `obs["cell_type"]` doesn't exist or has unexpected values), we get a degenerate
# marker dict and every downstream F1 number is unreliable.

# %% Cell 10 — Marker-dict diagnostics
from genevector.benchmarks.spatial.eval import _discover_grafiti_markers

discovered = _discover_grafiti_markers(adata, gt)
print("Grafiti markers discovered per cell_type:")
for ct, ms in discovered.items():
    print(f"  {ct:10s} ({len(ms):2d}): {ms}")

print()
print("Full marker dict for phenotype_probability:")
diag_prefixes = ("LIG_", "REC_", "NICHE_", "TRARE_", "HK_")
for pheno, ms in markers.items():
    n_grafiti = sum(1 for m in ms if not m.startswith(diag_prefixes))
    n_diag = sum(1 for m in ms if m.startswith(diag_prefixes))
    flag = "  ← grafiti=0, F1 unreliable" if n_grafiti == 0 else ""
    print(f"  {pheno:18s} grafiti={n_grafiti:2d} diag={n_diag:2d}{flag}")
    print(f"    -> {ms}")


# %% [markdown]
# # Scratch — paste experiments here.
#
# Some directions worth exploring:
#   - Try alternative phenotype markers (e.g. drop the shared T markers, keep only LIG/REC/NICHE/TRARE)
#     and re-run F1. If F1[T_rare] jumps from 0 to >0, the issue is marker dominance.
#   - Try `method="softmax"` vs `"normalized_exponential"` with different temperatures.
#     graph_xcorr embeddings may have a different similarity scale than mi.
#   - Compute paracrine AUC at *multiple training checkpoints* (currently we only have epoch=3000).
#     Maybe the embedding has the right structure at epoch 500 and over-trains by 3000.
#   - Look at gene-pair similarities not via cosine but via Euclidean distance after normalizing.
#     graph_xcorr's loss is MSE on raw scores, not cosine, so the geometry of the embedding may
#     be better captured by Euclidean than cosine.

# %% Cell 11 — Free-form scratch space
