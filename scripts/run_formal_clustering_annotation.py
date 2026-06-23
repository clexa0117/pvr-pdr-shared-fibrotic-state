import csv
import gzip
from collections import defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/formal_qc/filtered_matrices"
QC = ROOT / "results/formal_qc/cell_metrics"
OUT = ROOT / "results/formal_clustering"

COHORTS = {
    "GSE294329": ["GSM8902421_XZH02", "GSM8902422_WHN03", "GSM8902423_ZWB04", "GSM8902424_NPC05", "GSM8902425_JT06"],
    "GSE245561": ["GSM7845000_AF-TD-FAW-1", "GSM7845001_AF-TD-FAW-2", "GSM7845002_AF-TD-TED-2", "GSM7845003_AF-TTF-1"],
    "GSE165784": ["GSM5049904_RRD-ERM1", "GSM5049905_PDR-ERM-51", "GSM5049906_PDR-ERM-59", "GSM5277737_F47-PDR-ERM", "GSM5690478_PDR-FM-0609", "GSM5690479_PDR-ERM-210630"],
}

CURATED_ANNOTATIONS = {
    "GSE294329": {
        "0": "T_NK", "1": "Photoreceptor", "2": "Muller_Glial", "3": "Myeloid",
        "4": "T_NK", "5": "Myeloid", "6": "Fibroblast_Mesenchymal",
        "7": "RPE_like", "8": "Photoreceptor", "9": "Photoreceptor", "10": "Photoreceptor",
    },
    "GSE245561": {
        "0": "Endothelial", "1": "Pericyte", "2": "Myeloid",
        "3": "Fibroblast_Mesenchymal", "4": "T_NK", "5": "Endothelial",
        "6": "Myeloid", "7": "Myeloid", "8": "Erythroid",
    },
    "GSE165784": {
        "0": "Myeloid", "1": "Myeloid", "2": "Myeloid",
        "3": "Fibroblast_Mesenchymal", "4": "Myeloid", "5": "Myeloid",
        "6": "Cycling_Myeloid", "7": "T_NK", "8": "Endothelial",
    },
}

MARKERS = {
    "Fibroblast_Mesenchymal": {"COL1A1", "COL1A2", "COL3A1", "COL5A1", "DCN", "LUM", "COL6A1", "COL6A2", "FN1", "SPARC", "CTHRC1", "POSTN"},
    "Pericyte": {"RGS5", "PDGFRB", "CSPG4", "MCAM", "NOTCH3", "DES", "COL4A1", "COL4A2"},
    "Endothelial": {"PECAM1", "VWF", "KDR", "EMCN", "CLDN5", "ESAM", "ENG", "RAMP2"},
    "Myeloid": {"PTPRC", "TYROBP", "LST1", "FCER1G", "AIF1", "CTSS", "C1QA", "C1QB"},
    "T_NK": {"PTPRC", "CD3D", "CD3E", "TRAC", "NKG7", "CCL5", "GZMA", "KLRD1"},
    "RPE_like": {"RPE65", "BEST1", "MITF", "PMEL", "TYR", "TYRP1", "KRT8", "KRT18", "KRT19", "EPCAM"},
    "Muller_Glial": {"RLBP1", "GLUL", "SLC1A3", "GFAP", "AQP4", "CLU", "S100B", "FABP7"},
    "Photoreceptor": {"RCVRN", "GNAT1", "GNAT2", "RHO", "PDE6A", "PDE6B", "PDE6H", "NRL", "AIPL1", "GUCA1A", "GUCA1B"},
    "Cycling": {"MKI67", "TOP2A", "NUSAP1", "STMN1", "CENPF", "PTTG1"},
    "Mast": {"TPSAB1", "TPSB2", "KIT", "MS4A2", "CPA3"},
    "Erythroid": {"HBB", "HBA1", "HBA2", "ALAS2"},
}

INCOMPATIBLE = {
    frozenset(("Myeloid", "Endothelial")),
    frozenset(("Myeloid", "RPE_like")),
    frozenset(("Myeloid", "Muller_Glial")),
    frozenset(("Myeloid", "Fibroblast_Mesenchymal")),
    frozenset(("T_NK", "Endothelial")),
    frozenset(("T_NK", "RPE_like")),
    frozenset(("T_NK", "Muller_Glial")),
    frozenset(("T_NK", "Fibroblast_Mesenchymal")),
    frozenset(("Endothelial", "RPE_like")),
    frozenset(("Endothelial", "Muller_Glial")),
    frozenset(("Photoreceptor", "Myeloid")),
    frozenset(("Photoreceptor", "T_NK")),
    frozenset(("Photoreceptor", "Endothelial")),
}


def read_vector(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def load_cohort(samples):
    gene_lists = {sample: read_vector(INPUT / sample / "genes.tsv.gz") for sample in samples}
    common = set(gene_lists[samples[0]])
    for genes in gene_lists.values():
        common &= set(genes)
    common_genes = sorted(common)
    matrices, meta = [], []
    for sample in samples:
        matrix = sparse.load_npz(INPUT / sample / "counts_gene_by_cell.npz").tocsr()
        barcodes = read_vector(INPUT / sample / "barcodes.tsv.gz")
        gene_index = {gene: index for index, gene in enumerate(gene_lists[sample])}
        matrix = matrix[[gene_index[gene] for gene in common_genes]].T.tocsr()
        matrices.append(matrix)
        qc = pd.read_csv(QC / f"{sample}.csv.gz").set_index("barcode").loc[barcodes]
        meta.append(pd.DataFrame({
            "sample": sample,
            "barcode": barcodes,
            "total_umi": qc["total_umi"].to_numpy(),
            "detected_genes": qc["detected_genes"].to_numpy(),
            "mito_pct": qc["mito_pct"].to_numpy(),
            "scrublet_score": qc["scrublet_score"].to_numpy(),
        }))
    return sparse.vstack(matrices).tocsr(), common_genes, pd.concat(meta, ignore_index=True)


def normalize_log(counts):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scaled = counts.multiply(np.divide(10000, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)[:, None])
    scaled.data = np.log1p(scaled.data)
    return scaled.tocsr()


def select_hvg(lognorm, genes, n=2500):
    mean = np.asarray(lognorm.mean(axis=0)).ravel()
    mean_sq = np.asarray(lognorm.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean ** 2, 0)
    dispersion = variance / np.maximum(mean, 1e-8)
    exclude = np.fromiter((gene.upper().startswith(("MT-", "RPS", "RPL")) for gene in genes), bool)
    eligible = np.flatnonzero((mean > 0.01) & ~exclude)
    selected = eligible[np.argsort(dispersion[eligible])[-min(n, len(eligible)):]]
    return selected


def batch_center(pcs, samples):
    corrected = pcs.copy()
    global_mean = pcs.mean(axis=0)
    for sample in np.unique(samples):
        mask = samples == sample
        corrected[mask] = pcs[mask] - pcs[mask].mean(axis=0) + global_mean
    return corrected


def build_graph(pcs, samples, k_per_sample=4):
    edges, weights = {}, {}
    all_distances = []
    for sample in np.unique(samples):
        target = np.flatnonzero(samples == sample)
        n_neighbors = min(k_per_sample + 1, len(target))
        model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(pcs[target])
        distances, indices = model.kneighbors(pcs)
        for cell in range(len(pcs)):
            for local_neighbor, distance in zip(indices[cell], distances[cell]):
                neighbor = int(target[local_neighbor])
                if neighbor == cell:
                    continue
                edge = tuple(sorted((cell, neighbor)))
                edges[edge] = min(edges.get(edge, np.inf), float(distance))
                all_distances.append(distance)
    scale = np.median(all_distances) + 1e-8
    weighted_edges = {edge: float(np.exp(-distance / scale)) for edge, distance in edges.items()}
    graph = ig.Graph(n=len(pcs), edges=list(edges), directed=False)
    graph.es["weight"] = list(weighted_edges.values())
    return graph


def marker_scores(lognorm, genes):
    index = {gene: i for i, gene in enumerate(genes)}
    scores = {}
    detected = {}
    for label, marker_set in MARKERS.items():
        columns = [index[gene] for gene in marker_set if gene in index]
        scores[label] = np.asarray(lognorm[:, columns].mean(axis=1)).ravel() if columns else np.zeros(lognorm.shape[0])
        detected[label] = np.asarray((lognorm[:, columns] > 0).sum(axis=1)).ravel() if columns else np.zeros(lognorm.shape[0])
    return pd.DataFrame(scores), pd.DataFrame(detected)


def annotate_clusters(accession, clusters, scores, detected):
    rows = []
    labels = np.empty(len(clusters), dtype=object)
    for cluster in sorted(np.unique(clusters), key=int):
        mask = clusters == cluster
        medians = scores.loc[mask].median()
        positives = (detected.loc[mask] > 0).mean()
        ranking = medians.rank(pct=True) + positives.rank(pct=True)
        label = ranking.idxmax()
        if positives[label] < 0.20:
            label = "Unresolved"
        curated = CURATED_ANNOTATIONS[accession].get(cluster, label)
        labels[mask] = curated
        row = {"cluster": cluster, "cells": int(mask.sum()), "automatic_annotation": label, "annotation": curated}
        for name in MARKERS:
            row[f"{name}_median_score"] = medians[name]
            row[f"{name}_positive_pct"] = positives[name] * 100
        rows.append(row)
    return labels, pd.DataFrame(rows)


def second_doublet_flags(meta, scores, detected):
    high = pd.DataFrame(index=scores.index)
    for label in MARKERS:
        high[label] = (scores[label] >= scores[label].quantile(0.95)) & (detected[label] >= 2)
    incompatible_mix = np.zeros(len(meta), dtype=bool)
    pairs = np.full(len(meta), "", dtype=object)
    for pair in INCOMPATIBLE:
        left, right = tuple(pair)
        mask = high[left].to_numpy() & high[right].to_numpy()
        incompatible_mix |= mask
        pairs[mask] = left + "+" + right
    high_complexity = meta["detected_genes"].to_numpy() >= meta["detected_genes"].quantile(0.90)
    scrublet_support = meta["scrublet_score"].to_numpy() >= meta["scrublet_score"].quantile(0.90)
    flag = incompatible_mix & (high_complexity | scrublet_support)
    return flag, pairs


def cluster_markers(lognorm, genes, clusters, top_n=20):
    rows = []
    for cluster in sorted(np.unique(clusters), key=int):
        inside = clusters == cluster
        outside = ~inside
        mean_in = np.asarray(lognorm[inside].mean(axis=0)).ravel()
        mean_out = np.asarray(lognorm[outside].mean(axis=0)).ravel()
        pct_in = np.asarray((lognorm[inside] > 0).mean(axis=0)).ravel()
        pct_out = np.asarray((lognorm[outside] > 0).mean(axis=0)).ravel()
        score = (mean_in - mean_out) * np.maximum(pct_in - pct_out, 0)
        for index in np.argsort(score)[-top_n:][::-1]:
            rows.append({
                "cluster": cluster, "gene": genes[index], "marker_score": score[index],
                "mean_logexpr_in": mean_in[index], "mean_logexpr_out": mean_out[index],
                "pct_in": pct_in[index] * 100, "pct_out": pct_out[index] * 100,
            })
    return pd.DataFrame(rows)


def plot_umap(accession, embedding, meta):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, column, title in zip(axes, ["cluster", "annotation", "sample"], ["Leiden cluster", "Initial annotation", "Sample"]):
        values = meta[column].astype(str)
        categories = sorted(values.unique())
        palette = dict(zip(categories, sns.color_palette("husl", len(categories))))
        for category in categories:
            mask = values == category
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=2, alpha=0.65, color=palette[category], label=category)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=4, fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / accession / "umap_overview.png", dpi=180)
    plt.close(fig)


def process(accession, samples):
    cohort_out = OUT / accession
    cohort_out.mkdir(parents=True, exist_ok=True)
    counts, genes, meta = load_cohort(samples)
    lognorm = normalize_log(counts)
    hvg = select_hvg(lognorm, genes)
    pcs = TruncatedSVD(n_components=30, random_state=0).fit_transform(lognorm[:, hvg])
    corrected = batch_center(pcs, meta["sample"].to_numpy())
    graph = build_graph(corrected, meta["sample"].to_numpy())
    clusters = np.asarray(leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=0.7, seed=0
    ).membership).astype(str)
    embedding = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="euclidean", random_state=0).fit_transform(corrected)
    scores, detected = marker_scores(lognorm, genes)
    annotation, cluster_summary = annotate_clusters(accession, clusters, scores, detected)
    flags, pairs = second_doublet_flags(meta, scores, detected)
    meta["cluster"] = clusters
    meta["annotation"] = annotation
    meta["second_doublet_candidate"] = flags
    meta["incompatible_lineage_pair"] = pairs
    meta["final_singlet"] = ~flags
    meta["final_annotation"] = np.where(flags, "Second_round_doublet", annotation)
    meta["umap_1"] = embedding[:, 0]
    meta["umap_2"] = embedding[:, 1]
    for label in MARKERS:
        meta[f"{label}_score"] = scores[label]
        meta[f"{label}_detected_markers"] = detected[label]
    meta.to_csv(cohort_out / "cell_annotations.csv.gz", index=False)
    meta.loc[meta["final_singlet"], ["sample", "barcode", "cluster", "final_annotation"]].to_csv(
        cohort_out / "final_singlet_annotations.csv.gz", index=False
    )
    cluster_summary.to_csv(cohort_out / "cluster_annotation_summary.csv", index=False)
    cluster_markers(lognorm, genes, clusters).to_csv(cohort_out / "cluster_marker_genes.csv", index=False)
    plot_umap(accession, embedding, meta)
    return {
        "accession": accession,
        "samples": len(samples),
        "cells": len(meta),
        "common_genes": len(genes),
        "hvg": len(hvg),
        "clusters": meta["cluster"].nunique(),
        "second_doublet_candidates": int(flags.sum()),
        "second_doublet_candidate_pct": float(flags.mean() * 100),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for accession, samples in COHORTS.items():
        print(f"Processing {accession}", flush=True)
        rows.append(process(accession, samples))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "cohort_clustering_summary.csv", index=False)
    annotations = []
    for accession in COHORTS:
        cell = pd.read_csv(OUT / accession / "cell_annotations.csv.gz", usecols=["sample", "final_annotation", "second_doublet_candidate"])
        group = cell.groupby(["sample", "final_annotation"], as_index=False).agg(cells=("final_annotation", "size"), second_doublet_candidates=("second_doublet_candidate", "sum"))
        group = group.rename(columns={"final_annotation": "annotation"})
        group.insert(0, "accession", accession)
        annotations.append(group)
    pd.concat(annotations, ignore_index=True).to_csv(OUT / "sample_annotation_counts.csv", index=False)


if __name__ == "__main__":
    main()
