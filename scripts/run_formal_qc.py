import csv
import gzip
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scrublet as scr
from scipy import io, sparse


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "results/formal_qc"


def sample_list():
    rows = []
    for accession, disease in (("GSE294329", "PVR"), ("GSE245561", "PDR")):
        for matrix in sorted((RAW / accession / "files").glob("*_matrix.mtx.gz")):
            rows.append(
                (
                    accession,
                    disease,
                    matrix.name.removesuffix("_matrix.mtx.gz"),
                    "mtx_gz",
                    matrix,
                )
            )
    for path in sorted((RAW / "GSE165784/files").glob("*_matrix.tsv.gz")):
        disease = "PVR" if "RRD-ERM1" in path.name else "PDR"
        rows.append(
            (
                "GSE165784",
                disease,
                path.name.removesuffix("_matrix.tsv.gz"),
                "dense",
                path,
            )
        )
    for gsm, folder in (
        ("GSM5690478_PDR-FM-0609", "GSM5690478/PDR-FM-0609_matrix_10X"),
        ("GSM5690479_PDR-ERM-210630", "GSM5690479/PDR-ERM-210630_matrix_10X"),
    ):
        rows.append(
            (
                "GSE165784",
                "PDR",
                gsm,
                "mtx_plain",
                RAW / "GSE165784/nested" / folder / "matrix.mtx",
            )
        )
    return rows


def read_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t") for line in handle]


def read_mtx(fmt, matrix):
    if fmt == "mtx_gz":
        prefix = str(matrix).removesuffix("_matrix.mtx.gz")
        features = Path(prefix + "_features.tsv.gz")
        barcodes = Path(prefix + "_barcodes.tsv.gz")
        with gzip.open(matrix, "rb") as handle:
            counts = io.mmread(handle).tocsr()
    else:
        features = matrix.parent / "genes.tsv"
        barcodes = matrix.parent / "barcodes.tsv"
        # Explicit file handles avoid scipy's path parser issue with non-ASCII paths.
        with matrix.open("rb") as handle:
            counts = io.mmread(handle).tocsr()
    feature_rows = read_lines(features)
    genes = [row[1] if len(row) > 1 else row[0] for row in feature_rows]
    barcode_names = [row[0] for row in read_lines(barcodes)]
    return counts, genes, barcode_names


def read_dense(path):
    genes = []
    row_parts = []
    col_parts = []
    value_parts = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        barcodes = handle.readline().rstrip("\n").split("\t")[1:]
        for gene_index, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            genes.append(parts[0])
            values = np.fromiter(
                (int(float(value)) for value in parts[1:]),
                dtype=np.int32,
                count=len(barcodes),
            )
            nonzero = np.flatnonzero(values)
            if nonzero.size:
                row_parts.append(np.full(nonzero.size, gene_index, dtype=np.int32))
                col_parts.append(nonzero.astype(np.int32))
                value_parts.append(values[nonzero])
    counts = sparse.csr_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(len(genes), len(barcodes)),
    )
    return counts, genes, barcodes


def mad(values):
    median = np.median(values)
    return median, 1.4826 * np.median(np.abs(values - median))


def adaptive_limits(total, genes, mito_pct):
    positive = (total > 0) & (genes > 0)
    log_total_median, log_total_mad = mad(np.log10(total[positive]))
    log_gene_median, log_gene_mad = mad(np.log10(genes[positive]))
    mito_median, mito_mad = mad(mito_pct[positive])
    return {
        "min_umi": max(500, math.floor(10 ** (log_total_median - 3 * log_total_mad))),
        "max_umi": math.ceil(10 ** (log_total_median + 3 * log_total_mad)),
        "min_genes": max(200, math.floor(10 ** (log_gene_median - 3 * log_gene_mad))),
        "max_genes": math.ceil(10 ** (log_gene_median + 3 * log_gene_mad)),
        "max_mito_pct": min(25.0, max(10.0, mito_median + 3 * mito_mad)),
    }


def expected_doublet_rate(cell_count):
    return min(0.10, max(0.01, 0.008 * cell_count / 1000))


def run_scrublet(cell_by_gene, expected_rate):
    detector = scr.Scrublet(
        cell_by_gene,
        expected_doublet_rate=expected_rate,
        random_state=0,
    )
    n_components = min(30, max(5, cell_by_gene.shape[0] // 20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, predicted = detector.scrub_doublets(
            min_counts=2,
            min_cells=3,
            min_gene_variability_pctl=85,
            n_prin_comps=n_components,
            verbose=False,
        )
    return scores, predicted, float(detector.threshold_)


def write_cell_metrics(path, barcodes, metrics):
    fieldnames = ["barcode"] + list(metrics)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, barcode in enumerate(barcodes):
            row = {"barcode": barcode}
            for name, values in metrics.items():
                value = values[index]
                row[name] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)


def write_vector(path, values):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for value in values:
            handle.write(f"{value}\n")


def plot_qc(path, total, genes, mito_pct, scrublet_score, final_pass, limits, threshold):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].hist(np.log10(total[total > 0]), bins=60, color="#4c78a8")
    axes[0, 0].axvline(np.log10(limits["min_umi"]), color="red", linestyle="--")
    axes[0, 0].axvline(np.log10(limits["max_umi"]), color="red", linestyle="--")
    axes[0, 0].set_title("log10 UMI")
    axes[0, 1].hist(np.log10(genes[genes > 0]), bins=60, color="#59a14f")
    axes[0, 1].axvline(np.log10(limits["min_genes"]), color="red", linestyle="--")
    axes[0, 1].axvline(np.log10(limits["max_genes"]), color="red", linestyle="--")
    axes[0, 1].set_title("log10 detected genes")
    axes[1, 0].scatter(
        np.log10(total + 1),
        mito_pct,
        c=np.where(final_pass, "#59a14f", "#bab0ac"),
        s=3,
        alpha=0.5,
    )
    axes[1, 0].axhline(limits["max_mito_pct"], color="red", linestyle="--")
    axes[1, 0].set_xlabel("log10 UMI")
    axes[1, 0].set_ylabel("mitochondrial percent")
    axes[1, 1].hist(scrublet_score[np.isfinite(scrublet_score)], bins=60, color="#f28e2b")
    axes[1, 1].axvline(threshold, color="red", linestyle="--")
    axes[1, 1].set_title("Scrublet doublet score")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def process_sample(accession, disease, sample, fmt, path):
    counts, genes, barcodes = read_dense(path) if fmt == "dense" else read_mtx(fmt, path)
    counts = counts.tocsr()
    total = np.asarray(counts.sum(axis=0)).ravel().astype(np.int64)
    detected_genes = np.diff(counts.tocsc().indptr).astype(np.int32)
    mito_mask = np.fromiter((gene.upper().startswith("MT-") for gene in genes), dtype=bool)
    mito = np.asarray(counts[mito_mask].sum(axis=0)).ravel()
    mito_pct = np.divide(mito * 100, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    limits = adaptive_limits(total, detected_genes, mito_pct)

    low_quality_pass = (
        (total >= limits["min_umi"])
        & (detected_genes >= limits["min_genes"])
        & (mito_pct <= limits["max_mito_pct"])
    )
    upper_bound_pass = (total <= limits["max_umi"]) & (detected_genes <= limits["max_genes"])
    scrublet_input = counts[:, low_quality_pass].T.tocsr()
    rate = expected_doublet_rate(scrublet_input.shape[0])
    scores, predicted, threshold = run_scrublet(scrublet_input, rate)
    scrublet_score = np.full(counts.shape[1], np.nan)
    predicted_doublet = np.zeros(counts.shape[1], dtype=bool)
    scrublet_score[low_quality_pass] = scores
    predicted_doublet[low_quality_pass] = predicted
    final_pass = low_quality_pass & upper_bound_pass & ~predicted_doublet

    sample_out = OUT / "filtered_matrices" / sample
    sample_out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(sample_out / "counts_gene_by_cell.npz", counts[:, final_pass])
    write_vector(sample_out / "genes.tsv.gz", genes)
    write_vector(sample_out / "barcodes.tsv.gz", np.asarray(barcodes)[final_pass])

    fail_reason = np.full(counts.shape[1], "pass", dtype=object)
    fail_reason[~low_quality_pass] = "low_quality"
    fail_reason[low_quality_pass & ~upper_bound_pass] = "upper_outlier"
    fail_reason[low_quality_pass & upper_bound_pass & predicted_doublet] = "predicted_doublet"
    write_cell_metrics(
        OUT / "cell_metrics" / f"{sample}.csv.gz",
        barcodes,
        {
            "total_umi": total,
            "detected_genes": detected_genes,
            "mito_pct": mito_pct,
            "low_quality_pass": low_quality_pass,
            "upper_bound_pass": upper_bound_pass,
            "scrublet_score": scrublet_score,
            "predicted_doublet": predicted_doublet,
            "final_pass": final_pass,
            "fail_reason": fail_reason,
        },
    )
    plot_qc(
        OUT / "plots" / f"{sample}.png",
        total,
        detected_genes,
        mito_pct,
        scrublet_score,
        final_pass,
        limits,
        threshold,
    )
    return {
        "accession": accession,
        "disease": disease,
        "sample": sample,
        "raw_cells": counts.shape[1],
        **limits,
        "low_quality_pass_cells": int(low_quality_pass.sum()),
        "upper_outlier_cells": int((low_quality_pass & ~upper_bound_pass).sum()),
        "scrublet_input_cells": scrublet_input.shape[0],
        "expected_doublet_rate": rate,
        "scrublet_threshold": threshold,
        "predicted_doublets": int((low_quality_pass & predicted_doublet).sum()),
        "final_pass_cells": int(final_pass.sum()),
        "final_pass_pct": float(final_pass.mean() * 100),
        "median_umi_final": float(np.median(total[final_pass])),
        "median_genes_final": float(np.median(detected_genes[final_pass])),
        "median_mito_pct_final": float(np.median(mito_pct[final_pass])),
    }


def write_summary(rows):
    with (OUT / "sample_qc_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    total_raw = sum(row["raw_cells"] for row in rows)
    total_final = sum(row["final_pass_cells"] for row in rows)
    lines = [
        "# Formal single-cell QC summary",
        "",
        f"- Samples processed: {len(rows)}",
        f"- Raw cells: {total_raw}",
        f"- Final retained singlets: {total_final} ({total_final / total_raw * 100:.1f}%)",
        f"- Predicted doublets removed: {sum(row['predicted_doublets'] for row in rows)}",
        f"- Upper-outlier cells removed: {sum(row['upper_outlier_cells'] for row in rows)}",
        "",
        "| Accession | Disease | Sample | Raw | Final singlets | Retained | Doublets |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['accession']} | {row['disease']} | {row['sample']} | "
        f"{row['raw_cells']} | {row['final_pass_cells']} | "
        f"{row['final_pass_pct']:.1f}% | {row['predicted_doublets']} |"
        for row in rows
    )
    (OUT / "formal_qc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    (OUT / "cell_metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in sample_list():
        print(f"Processing {sample[2]}", flush=True)
        rows.append(process_sample(*sample))
    write_summary(rows)


if __name__ == "__main__":
    main()
