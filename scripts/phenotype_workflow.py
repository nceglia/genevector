#!/usr/bin/env python
"""Probabilistic cell-type phenotyping workflow (spatial + scRNA).

Generalised genevector phenotyping driven by a marker dictionary. Given an AnnData, a
JSON ``{phenotype: [marker genes]}`` dict, and an output directory, it:

  1. cleans the panel (optional mito/ribo removal),
  2. selects genes = HVG ∪ SVG (Moran's I via grafiti when spatial coords + grafiti are
     available; otherwise scanpy HVG), always force-keeping the marker genes,
  3. QCs the marker dict (flags absent / low-specificity / off-target markers),
  4. trains a genevector embedding — using the spatially-aware ``graph_mi`` target for
     spatial data (neighbour aggregation denoises sparse counts before estimating MI),
     and plain ``mi`` for dissociated data,
  5. builds the cell embedding (optionally count-adaptive spatial denoising of cell
     vectors, opt-in via --denoise),
  6. assigns phenotypes via ``phenotype_probability`` (optional --debias / --contrastive /
     --label-prop), and
  7. writes the annotated AnnData, the marker-QC table, a per-cell assignment table and a
     run summary into the output directory.

Example
-------
    python scripts/phenotype_workflow.py \
        --input data.h5ad --markers markers.json --output out/ \
        --target auto --denoise --device cuda

Marker JSON format::

    {"Tumor": ["EPCAM", "KRT8"], "T cell": ["CD3D", "CD3E"], ...}
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

try:
    from genevector.data import GeneVectorDataset
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Gene panel cleaning ───────────────────────────────────────

def clean_panel(adata, keep):
    """Drop mito/ribo/ambiguous genes (keeping any in `keep`)."""
    keepU = {g.upper() for g in keep}
    genes = []
    for g in adata.var.index:
        gu = str(g).upper()
        if gu in keepU:
            genes.append(g)
            continue
        if gu.startswith(("MT-", "RPS", "RPL")):
            continue
        if "." in gu:
            continue
        if "-" in gu and "HLA" not in gu:
            continue
        genes.append(g)
    n_removed = adata.n_vars - len(genes)
    return adata[:, genes].copy(), n_removed


# ─── Spatial graph ─────────────────────────────────────────────

def build_spatial_graph(adata, spatial_key, k):
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    W = kneighbors_graph(coords, n_neighbors=min(k, adata.n_obs - 1),
                         mode="connectivity", include_self=False)
    return ((W + W.T) > 0).astype(float).tocsr()


# ─── Gene selection: HVG ∪ SVG ─────────────────────────────────

def select_genes(adata, n_top_genes, marker_genes, spatial, batch_key, counts_layer,
                 grafiti_path=None):
    """Return HVG ∪ SVG (force-keeping marker_genes), upper-cased to match genevector."""
    keep = {g.upper() for g in marker_genes}
    layer = counts_layer if counts_layer in adata.layers else None
    svg_ok = False
    if spatial:
        if grafiti_path and grafiti_path not in sys.path:
            sys.path.insert(0, grafiti_path)
        try:
            import grafiti as gf
            if "sample_fov" not in adata.obs:
                adata.obs["sample_fov"] = (adata.obs[batch_key].astype(str)
                                           if batch_key and batch_key in adata.obs else "s0")
            df = gf.pp.spatially_variable_genes(
                adata, layer=layer, hvg_layer=layer, n_top_genes=n_top_genes,
                union_hvg=True, white_list=list(marker_genes), inplace=False)
            flag = df.get("spatially_variable")
            sel = (df.index[flag.astype(bool)].tolist() if flag is not None
                   else df.sort_values(df.columns[0], ascending=False).head(n_top_genes).index.tolist())
            svg_ok = True
            log(f"Selected {len(sel)} genes via grafiti HVG∪SVG (Moran's I).")
        except Exception as e:
            log(f"grafiti SVG unavailable ({e}); falling back to scanpy HVG.")
    if not svg_ok:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3",
                                        layer=layer, batch_key=batch_key if batch_key in adata.obs else None)
            sel = adata.var.index[adata.var["highly_variable"]].tolist()
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            sel = adata.var.index[adata.var["highly_variable"]].tolist()
        log(f"Selected {len(sel)} HVGs (scanpy).")
    selU = {str(g).upper() for g in sel} | keep
    return [g for g in adata.var.index if str(g).upper() in selU]


# ─── Main ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Probabilistic cell-type phenotyping (spatial + scRNA).")
    p.add_argument("--input", required=True, help="Input .h5ad")
    p.add_argument("--markers", required=True, help="JSON {phenotype: [marker genes]}")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--target", default="auto", choices=["auto", "mi", "graph_mi", "graph_cross_mi"],
                   help="Training target. 'auto' = graph_mi when spatial else mi.")
    p.add_argument("--n-genes", type=int, default=2000, help="HVG∪SVG count.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--spatial-key", default="spatial", help="obsm key for coordinates.")
    p.add_argument("--knn", type=int, default=10, help="k for the spatial graph.")
    p.add_argument("--batch-key", default=None, help="obs column for HVG batch / sample_fov.")
    p.add_argument("--counts-layer", default="counts", help="layer holding raw counts.")
    p.add_argument("--no-clean", action="store_true", help="skip mito/ribo gene removal.")
    p.add_argument("--denoise", action="store_true",
                   help="count-adaptive spatial denoising of cell vectors (spatial only).")
    p.add_argument("--debias", type=float, default=0.0,
                   help="fraction of dataset vector subtracted before scoring (0=off).")
    p.add_argument("--contrastive", action="store_true",
                   help="subtract competing-phenotype means before scoring.")
    p.add_argument("--score-norm", default="none", choices=["none", "zscore", "rank"],
                   help="per-phenotype normalization of similarity columns before assignment.")
    p.add_argument("--label-prop", type=float, default=0.0,
                   help="spatial label-propagation coupling on probabilities (0=off, spatial only).")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--mi-backend", default="auto")
    p.add_argument("--grafiti-path", default=None, help="path to a grafiti checkout (for SVG).")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    log(f"Device: {device}")

    adata = sc.read_h5ad(args.input)
    log(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes.")
    markers = json.load(open(args.markers))
    markers = {k: [str(g).upper() for g in v] for k, v in markers.items()}
    marker_genes = sorted({g for v in markers.values() for g in v})
    log(f"{len(markers)} phenotypes, {len(marker_genes)} unique markers.")

    spatial = args.spatial_key in adata.obsm
    log(f"Spatial coordinates: {'found' if spatial else 'not found'} "
        f"(obsm['{args.spatial_key}']).")

    # ensure raw counts in .X for MI discretization
    if args.counts_layer in adata.layers:
        adata.X = adata.layers[args.counts_layer].copy()
    else:
        adata.layers[args.counts_layer] = adata.X.copy()
        log(f"No '{args.counts_layer}' layer; using .X as counts.")

    if not args.no_clean:
        adata, n_removed = clean_panel(adata, marker_genes)
        log(f"Removed {n_removed} mito/ribo/ambiguous genes.")

    # gene selection (HVG ∪ SVG ∪ markers)
    sel = select_genes(adata, args.n_genes, marker_genes, spatial, args.batch_key,
                       args.counts_layer, args.grafiti_path)
    adata = adata[:, sel].copy()
    adata.X = adata.layers[args.counts_layer].copy()
    log(f"Training on {adata.n_vars} genes.")

    # marker QC (needs a CellEmbedding-less quick pass — reuse the static logic via a temp embed later)
    # build spatial graph
    target = args.target
    tkw = None
    W = None
    if spatial:
        W = build_spatial_graph(adata, args.spatial_key, args.knn)
        if target == "auto":
            target = "graph_mi"
        if target in ("graph_mi", "graph_cross_mi", "graph_xcorr"):
            tkw = {"graph": W}
        if target in ("graph_mi", "graph_cross_mi") and adata.n_vars > 2500 and device != "cuda":
            log(f"NOTE: {target} on {adata.n_vars} genes uses the vectorized torch kernel "
                f"(~20-35x faster than numpy); pass --device cuda to run it on GPU, or reduce "
                f"--n-genes for whole-transcriptome panels.")
    else:
        if target == "auto":
            target = "mi"
        if target.startswith("graph"):
            log("Graph target requested but no spatial coords; using 'mi'.")
            target = "mi"
    log(f"Target: {target}.")

    vec_path = os.path.join(args.output, "embedding.vec")
    ds = GeneVectorDataset(adata.copy(), load_expression=True, signed_mi=True, device=device,
                           target=target, target_kwargs=tkw, mi_backend=args.mi_backend,
                           use_cache=False)
    log("Training genevector...")
    gv = GeneVector(ds, output_file=vec_path, emb_dimension=args.dim, c=100, gain=10,
                    init_ortho=True, device=device)
    gv.train(args.epochs, update_interval=max(1, args.epochs // 5))

    embed = GeneEmbedding(vec_path, ds, vector="average")
    cembed = CellEmbedding(ds, embed)

    # marker QC report
    qc = cembed.qc_marker_dict(adata, markers, layer=args.counts_layer)
    qc.to_csv(os.path.join(args.output, "marker_qc.csv"), index=False)
    log(f"Marker QC -> marker_qc.csv ({int((qc['flag']!='ok').sum())} flagged of {len(qc)}).")

    # opt-in spatial denoising of cell vectors (before get_adata)
    if args.denoise and spatial:
        counts = np.asarray(adata.layers[args.counts_layer].sum(1)).ravel()
        # graph aligned to the cell-matrix order (== list(cembed.data.keys()))
        order = list(cembed.data.keys())
        pos = {c: i for i, c in enumerate(adata.obs.index)}
        idx = [pos[c] for c in order]
        Wsub = W[idx][:, idx]
        cembed.denoise_cell_vectors(Wsub, adaptive=True, counts=counts[idx])
        log("Applied count-adaptive spatial denoising to cell vectors.")

    adata_gv = cembed.get_adata()

    # label-prop graph aligned to the (possibly filtered) cell order
    lp_graph = None
    if args.label_prop > 0 and spatial:
        pos = {c: i for i, c in enumerate(adata.obs.index)}
        idx = [pos[c] for c in adata_gv.obs.index]
        lp_graph = W[idx][:, idx]

    log("Assigning phenotypes...")
    adata_gv = cembed.phenotype_probability(
        adata_gv, markers, method="normalized_exponential", temperature=args.temperature,
        target_col="genevector", debias=args.debias, contrastive=args.contrastive,
        score_norm=args.score_norm, lp_graph=lp_graph, lp_alpha=args.label_prop)

    # outputs
    out_h5ad = os.path.join(args.output, "phenotyped.h5ad")
    adata_gv.write_h5ad(out_h5ad)
    prob_cols = [c for c in adata_gv.obs.columns if "Pseudo-probability" in c]
    assign = adata_gv.obs[["genevector"] + prob_cols].copy()
    assign.to_csv(os.path.join(args.output, "assignments.csv"))
    summary = {
        "input": args.input, "n_cells": int(adata_gv.n_obs), "n_genes": int(adata.n_vars),
        "spatial": bool(spatial), "target": target, "denoise": bool(args.denoise and spatial),
        "debias": args.debias, "contrastive": bool(args.contrastive),
        "score_norm": args.score_norm, "label_prop": args.label_prop,
        "label_counts": {k: int(v) for k, v in adata_gv.obs["genevector"].value_counts().items()},
        "markers_flagged": int((qc["flag"] != "ok").sum()),
    }
    json.dump(summary, open(os.path.join(args.output, "summary.json"), "w"), indent=2)
    log(f"Done. Wrote phenotyped.h5ad, assignments.csv, marker_qc.csv, summary.json to {args.output}")
    log(f"Label counts: {summary['label_counts']}")


if __name__ == "__main__":
    main()
