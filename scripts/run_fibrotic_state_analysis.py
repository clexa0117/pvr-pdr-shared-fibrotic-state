import gzip
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
MATRIX_ROOT = ROOT / "results/formal_qc/filtered_matrices"
ANNOTATION_ROOT = ROOT / "results/formal_clustering"
OUT = ROOT / "results/fibrotic_state_analysis"

COHORTS = {
    "GSE294329": ["GSM8902421_XZH02", "GSM8902422_WHN03", "GSM8902423_ZWB04", "GSM8902424_NPC05", "GSM8902425_JT06"],
    "GSE245561": ["GSM7845000_AF-TD-FAW-1", "GSM7845001_AF-TD-FAW-2", "GSM7845002_AF-TD-TED-2", "GSM7845003_AF-TTF-1"],
    "GSE165784": ["GSM5049904_RRD-ERM1", "GSM5049905_PDR-ERM-51", "GSM5049906_PDR-ERM-59", "GSM5277737_F47-PDR-ERM", "GSM5690478_PDR-FM-0609", "GSM5690479_PDR-ERM-210630"],
}
TARGET_TYPES = {"Fibroblast_Mesenchymal", "RPE_like", "Muller_Glial", "Pericyte"}

PROGRAMS = {
    "ECM": {"COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3", "COL12A1", "FN1", "DCN", "LUM", "SPARC", "TGFBI", "POSTN", "CTHRC1"},
    "Contractile": {"ACTA2", "TAGLN", "MYL9", "TPM1", "TPM2", "CNN1", "CALD1", "MYLK"},
    "TGFB_Mechanosensing": {"TGFB1", "TGFBR1", "TGFBR2", "CTGF", "YAP1", "WWTR1", "ITGA5", "ITGB1", "THBS1"},
    "Inflammatory": {"IL1B", "TNF", "CXCL8", "CCL2", "NFKBIA", "ICAM1", "CXCL2", "CXCL3"},
    "Stress_ImmediateEarly": {"FOS", "FOSB", "JUN", "JUNB", "EGR1", "ATF3", "HSPA1A", "HSPA1B"},
    "RPE_origin": {"RPE65", "BEST1", "MITF", "PMEL", "TYR", "TYRP1", "KRT8", "KRT18", "KRT19", "EPCAM", "SERPINF1"},
    "Glial_origin": {"RLBP1", "GLUL", "SLC1A3", "GFAP", "AQP4", "CLU", "S100B", "FABP7"},
    "Pericyte_origin": {"RGS5", "PDGFRB", "CSPG4", "MCAM", "NOTCH3", "DES", "COL4A1", "COL4A2", "PRKG1"},
}
STATE_PROGRAMS = ["ECM", "Contractile", "TGFB_Mechanosensing", "Inflammatory", "Stress_ImmediateEarly"]
ORIGIN_PROGRAMS = ["RPE_origin", "Glial_origin", "Pericyte_origin"]
CURATED_STATE_LABELS = {
    "0": "Low_fibrosis_mixed",
    "1": "Shared_ECM_TGFB_high",
    "2": "Muller_stress",
    "3": "RPE_inflammatory",
    "4": "Pericyte_contractile",
}
EXCLUDE_FROM_STATE_HVG = set().union(*[PROGRAMS[name] for name in ORIGIN_PROGRAMS])
EXCLUDE_FROM_STATE_HVG |= {
    "PTPRC", "TYROBP", "LST1", "FCER1G", "C1QA", "C1QB", "C1QC",
    "PECAM1", "VWF", "KDR", "EMCN", "CLDN5", "RAMP2",
    "RCVRN", "GNAT1", "GNAT2", "RHO", "PDE6A", "PDE6B",
}


def read_vector(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def disease_for(sample):
    return "PVR" if sample.startswith("GSM890") or "RRD-ERM1" in sample else "PDR"


def load_target_cells():
    sample_annotations = {}
    gene_lists = {}
    for accession, samples in COHORTS.items():
        annotations = pd.read_csv(ANNOTATION_ROOT / accession / "final_singlet_annotations.csv.gz")
        for sample in samples:
            selected = annotations[(annotations["sample"] == sample) & annotations["final_annotation"].isin(TARGET_TYPES)].copy()
            sample_annotations[sample] = selected
            gene_lists[sample] = read_vector(MATRIX_ROOT / sample / "genes.tsv.gz")

    common = set(gene_lists[next(iter(gene_lists))])
    for genes in gene_lists.values():
        common &= set(genes)
    common_genes = sorted(common)

    matrices = []
    metadata = []
    for accession, samples in COHORTS.items():
        for sample in samples:
            selected = sample_annotations[sample]
            matrix = sparse.load_npz(MATRIX_ROOT / sample / "counts_gene_by_cell.npz").tocsr()
            barcodes = read_vector(MATRIX_ROOT / sample / "barcodes.tsv.gz")
            barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
            gene_index = {gene: index for index, gene in enumerate(gene_lists[sample])}
            cell_indices = [barcode_index[barcode] for barcode in selected["barcode"]]
            matrix = matrix[[gene_index[gene] for gene in common_genes]][:, cell_indices].T.tocsr()
            matrices.append(matrix)
            meta = selected[["sample", "barcode", "final_annotation"]].copy()
            meta.insert(0, "accession", accession)
            meta["disease"] = disease_for(sample)
            metadata.append(meta)
    return sparse.vstack(matrices).tocsr(), common_genes, pd.concat(metadata, ignore_index=True)


def normalize_log(counts):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scaled = counts.multiply(np.divide(10000, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)[:, None])
    scaled.data = np.log1p(scaled.data)
    return scaled.tocsr()


def program_scores(lognorm, genes):
    index = {gene: i for i, gene in enumerate(genes)}
    scores = {}
    detected = {}
    for name, marker_set in PROGRAMS.items():
        columns = [index[gene] for gene in marker_set if gene in index]
        scores[name] = np.asarray(lognorm[:, columns].mean(axis=1)).ravel()
        detected[name] = np.asarray((lognorm[:, columns] > 0).sum(axis=1)).ravel()
    return pd.DataFrame(scores), pd.DataFrame(detected)


def select_state_hvg(lognorm, genes, n=2000):
    mean = np.asarray(lognorm.mean(axis=0)).ravel()
    mean_sq = np.asarray(lognorm.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean ** 2, 0)
    dispersion = variance / np.maximum(mean, 1e-8)
    excluded = np.fromiter(
        (
            gene in EXCLUDE_FROM_STATE_HVG
            or gene.upper().startswith(("MT-", "RPS", "RPL"))
            for gene in genes
        ),
        bool,
    )
    eligible = np.flatnonzero((mean > 0.01) & ~excluded)
    return eligible[np.argsort(dispersion[eligible])[-min(n, len(eligible)):]]


def sample_center(values, samples):
    corrected = values.copy()
    global_mean = values.mean(axis=0)
    for sample in np.unique(samples):
        mask = samples == sample
        corrected[mask] = values[mask] - values[mask].mean(axis=0) + global_mean
    return corrected


def balanced_graph(values, samples, k_per_sample=2):
    edges = {}
    distances_all = []
    for sample in np.unique(samples):
        target = np.flatnonzero(samples == sample)
        neighbors = min(k_per_sample + 1, len(target))
        model = NearestNeighbors(n_neighbors=neighbors).fit(values[target])
        distances, indices = model.kneighbors(values)
        for cell in range(len(values)):
            for local_neighbor, distance in zip(indices[cell], distances[cell]):
                neighbor = int(target[local_neighbor])
                if cell == neighbor:
                    continue
                edge = tuple(sorted((cell, neighbor)))
                edges[edge] = min(edges.get(edge, np.inf), float(distance))
                distances_all.append(distance)
    scale = np.median(distances_all) + 1e-8
    graph = ig.Graph(n=len(values), edges=list(edges), directed=False)
    graph.es["weight"] = [float(np.exp(-distance / scale)) for distance in edges.values()]
    return graph


def summarize_clusters(meta, scores):
    rows = []
    for cluster in sorted(meta["state_cluster"].unique(), key=int):
        mask = meta["state_cluster"] == cluster
        row = {
            "state_cluster": cluster,
            "cells": int(mask.sum()),
            "PVR_cells": int((meta.loc[mask, "disease"] == "PVR").sum()),
            "PDR_cells": int((meta.loc[mask, "disease"] == "PDR").sum()),
            "PVR_patients": int(meta.loc[mask & (meta["disease"] == "PVR"), "sample"].nunique()),
            "PDR_patients": int(meta.loc[mask & (meta["disease"] == "PDR"), "sample"].nunique()),
            "patients": int(meta.loc[mask, "sample"].nunique()),
        }
        pvr_fraction = row["PVR_cells"] / row["cells"]
        accession_fraction = meta.loc[mask, "accession"].value_counts(normalize=True).max()
        row["PVR_fraction"] = pvr_fraction
        row["max_accession_fraction"] = accession_fraction
        row["shared_cross_disease"] = bool(
            row["PVR_patients"] >= 2
            and row["PDR_patients"] >= 2
            and 0.15 <= pvr_fraction <= 0.85
            and accession_fraction <= 0.75
        )
        for name in PROGRAMS:
            row[f"{name}_median"] = float(scores.loc[mask, name].median())
        for source in sorted(TARGET_TYPES):
            row[f"{source}_cells"] = int((meta.loc[mask, "final_annotation"] == source).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["state_annotation"] = summary["state_cluster"].map(CURATED_STATE_LABELS)
    labels = dict(zip(summary["state_cluster"], summary["state_annotation"]))
    return labels, summary


def cluster_markers(lognorm, genes, clusters, top_n=25):
    rows = []
    for cluster in sorted(np.unique(clusters), key=int):
        inside = clusters == cluster
        outside = ~inside
        mean_in = np.asarray(lognorm[inside].mean(axis=0)).ravel()
        mean_out = np.asarray(lognorm[outside].mean(axis=0)).ravel()
        pct_in = np.asarray((lognorm[inside] > 0).mean(axis=0)).ravel()
        pct_out = np.asarray((lognorm[outside] > 0).mean(axis=0)).ravel()
        statistic = (mean_in - mean_out) * np.maximum(pct_in - pct_out, 0)
        for index in np.argsort(statistic)[-top_n:][::-1]:
            rows.append({
                "state_cluster": cluster, "gene": genes[index], "marker_score": statistic[index],
                "mean_logexpr_in": mean_in[index], "mean_logexpr_out": mean_out[index],
                "pct_in": pct_in[index] * 100, "pct_out": pct_out[index] * 100,
            })
    return pd.DataFrame(rows)


def plot_umap(meta):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, column, title in zip(
        axes,
        ["state_annotation", "disease", "final_annotation", "sample"],
        ["Fibrotic state", "Disease", "Source-like major type", "Patient"],
    ):
        categories = sorted(meta[column].unique())
        palette = dict(zip(categories, sns.color_palette("husl", len(categories))))
        for category in categories:
            mask = meta[column] == category
            ax.scatter(meta.loc[mask, "umap_1"], meta.loc[mask, "umap_2"], s=3, alpha=0.65, color=palette[category], label=category)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=4, fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fibrotic_state_umap.png", dpi=180)
    plt.close(fig)


def plot_program_heatmap(summary):
    values = summary.set_index("state_cluster")[[f"{name}_median" for name in PROGRAMS]]
    values.columns = [column.removesuffix("_median") for column in values.columns]
    values = values.apply(lambda column: (column - column.mean()) / (column.std(ddof=0) or 1), axis=0)
    fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.45)))
    sns.heatmap(values, cmap="vlag", center=0, annot=True, fmt=".1f", ax=ax)
    ax.set_title("State-cluster program scores (column z-score)")
    fig.tight_layout()
    fig.savefig(OUT / "state_program_heatmap.png", dpi=180)
    plt.close(fig)


def patient_program_summary(meta, scores):
    merged = pd.concat([meta[["accession", "disease", "sample", "final_annotation"]], scores[STATE_PROGRAMS]], axis=1)
    return merged.groupby(["accession", "disease", "sample", "final_annotation"], as_index=False)[STATE_PROGRAMS].median()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts, genes, meta = load_target_cells()
    lognorm = normalize_log(counts)
    scores, detected = program_scores(lognorm, genes)
    hvg = select_state_hvg(lognorm, genes)
    pcs = TruncatedSVD(n_components=30, random_state=0).fit_transform(lognorm[:, hvg])
    state_features = np.column_stack([pcs, scores[STATE_PROGRAMS].to_numpy()])
    corrected = sample_center(state_features, meta["sample"].to_numpy())
    graph = balanced_graph(corrected, meta["sample"].to_numpy())
    clusters = np.asarray(
        leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=0.6, seed=0,
        ).membership
    ).astype(str)
    embedding = umap.UMAP(n_neighbors=25, min_dist=0.25, random_state=0).fit_transform(corrected)
    meta["state_cluster"] = clusters
    labels, cluster_summary = summarize_clusters(meta, scores)
    meta["state_annotation"] = [labels[cluster] for cluster in clusters]
    meta["umap_1"] = embedding[:, 0]
    meta["umap_2"] = embedding[:, 1]
    for name in PROGRAMS:
        meta[f"{name}_score"] = scores[name]
        meta[f"{name}_detected_genes"] = detected[name]

    meta.to_csv(OUT / "fibrotic_cell_states.csv.gz", index=False)
    cluster_summary.to_csv(OUT / "state_cluster_summary.csv", index=False)
    markers = cluster_markers(lognorm, genes, clusters)
    markers.to_csv(OUT / "state_cluster_marker_genes.csv", index=False)
    markers[markers["state_cluster"] == "1"].to_csv(OUT / "shared_state_marker_genes.csv", index=False)
    patient_program_summary(meta, scores).to_csv(OUT / "patient_source_program_summary.csv", index=False)

    patient_state = meta.groupby(["accession", "disease", "sample", "state_annotation"], as_index=False).size()
    patient_state.to_csv(OUT / "patient_state_counts.csv", index=False)
    state_source = meta.groupby(["state_annotation", "disease", "final_annotation"], as_index=False).size()
    state_source.to_csv(OUT / "state_source_disease_counts.csv", index=False)
    shared = meta[meta["state_annotation"] == "Shared_ECM_TGFB_high"]
    shared_patient = shared.groupby(["accession", "disease", "sample"], as_index=False).agg(
        cells=("barcode", "size"),
        ECM_median=("ECM_score", "median"),
        Contractile_median=("Contractile_score", "median"),
        TGFB_Mechanosensing_median=("TGFB_Mechanosensing_score", "median"),
        RPE_origin_median=("RPE_origin_score", "median"),
        Glial_origin_median=("Glial_origin_score", "median"),
        Pericyte_origin_median=("Pericyte_origin_score", "median"),
    )
    shared_patient.to_csv(OUT / "shared_state_patient_summary.csv", index=False)
    plot_umap(meta)
    plot_program_heatmap(cluster_summary)

    lines = [
        "# ", "",
        f"- : {len(meta)}",
        f"- : {meta['sample'].nunique()}/15",
        f"- PVR : {(meta['disease'] == 'PVR').sum()}",
        f"- PDR : {(meta['disease'] == 'PDR').sum()}",
        f"- : {meta['state_cluster'].nunique()}",
        f"- : {cluster_summary['shared_cross_disease'].sum()}",
        "",
        ": PVR  PDR  2 ,  15%,  GEO  75%.",
        "",
        "## ",
        "",
        "-  `Shared_ECM_TGFB_high`,  15 .",
        "-  `COL1A1`, `COL3A1`, `POSTN`, `FN1`, `SERPINE1`  `CTHRC1`.",
        "- PVR  `Muller_stress`  `RPE_inflammatory`, PDR  `Pericyte_contractile`.",
        "- /, , .",
        "",
        "| Cluster | State | Cells | PVR patients | PDR patients | Shared |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {row.state_cluster} | {row.state_annotation} | {row.cells} | {row.PVR_patients} | {row.PDR_patients} | {row.shared_cross_disease} |"
        for row in cluster_summary.itertuples()
    )
    (OUT / "fibrotic_state_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
