import gzip
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
PSEUDO = ROOT / "results/patient_pseudobulk"
STATE = ROOT / "results/fibrotic_state_analysis"
TARGET = ROOT / "results/target_cellular_specificity"
MATRIX_ROOT = ROOT / "results/formal_qc/filtered_matrices"
OUT = ROOT / "results/robustness_sensitivity"

FOCUS_TARGETS = ["POSTN", "CTHRC1", "SULF1"]
ALL_CANDIDATES = ["POSTN", "FN1", "THBS2", "CTHRC1", "AEBP1", "IGFBP7", "SULF1", "CFH", "TIMP1", "SERPINE1"]
SHARED_STATE = "Shared_ECM_TGFB_high"


def threshold_grid(cpm, marker_genes, thresholds, required_fractions, formal_core=None):
    marker_genes = sorted(set(marker_genes) & set(cpm.columns))
    formal_core = set(formal_core or [])
    rows = []
    for threshold in thresholds:
        detected = cpm[marker_genes].ge(threshold).sum(axis=0)
        for fraction in required_fractions:
            required = math.ceil(len(cpm) * fraction - 1e-12)
            qualifying = set(detected[detected >= required].index)
            rows.append({
                "cpm_threshold": threshold,
                "required_patient_fraction": fraction,
                "required_patients": required,
                "total_patients": len(cpm),
                "qualifying_marker_genes": len(qualifying),
                "formal_core_retained": len(qualifying & formal_core),
                "formal_core_total": len(formal_core),
                "formal_core_retention_fraction": len(qualifying & formal_core) / len(formal_core) if formal_core else np.nan,
                "qualifying_genes": ";".join(sorted(qualifying)),
            })
    return pd.DataFrame(rows)


def define_alternative_states(cells, top_fractions):
    ranked = cells.copy()
    ranked["ECM_percentile"] = ranked.groupby("sample")["ECM_score"].rank(pct=True, method="average")
    ranked["TGFB_percentile"] = ranked.groupby("sample")["TGFB_Mechanosensing_score"].rank(pct=True, method="average")
    ranked["combined_percentile"] = (ranked["ECM_percentile"] + ranked["TGFB_percentile"]) / 2
    outputs = []
    for fraction in top_fractions:
        current = ranked.copy()
        current["alternative_state"] = f"patient_top_{int(round(fraction * 100))}pct"
        selected_indices = []
        for _, group in current.groupby("sample"):
            n_select = max(1, math.ceil(len(group) * fraction))
            selected_indices.extend(group.nlargest(n_select, "combined_percentile").index)
        current["selected"] = current.index.isin(selected_indices)
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True)


def read_vector(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def load_cpm():
    counts = pd.read_csv(PSEUDO / "shared_state_pseudobulk_counts.csv.gz", index_col=0)
    return counts.div(counts.sum(axis=1), axis=0) * 1e6


def leave_one_out_core(cpm, metadata, formal_core, unit_column):
    rows = []
    for unit in sorted(metadata[unit_column].unique()):
        keep_samples = metadata.loc[metadata[unit_column] != unit, "sample"]
        subset = cpm.loc[keep_samples, formal_core]
        detected = subset.ge(1).sum(axis=0)
        retained = detected[detected == len(subset)].index.tolist()
        remaining_meta = metadata[metadata[unit_column] != unit]
        rows.append({
            f"excluded_{unit_column}": unit,
            "remaining_patients": len(subset),
            "remaining_PVR_patients": int((remaining_meta["disease"] == "PVR").sum()),
            "remaining_PDR_patients": int((remaining_meta["disease"] == "PDR").sum()),
            "formal_core_retained": len(retained),
            "formal_core_total": len(formal_core),
            "formal_core_retention_fraction": len(retained) / len(formal_core),
            "lost_formal_core_genes": ";".join(sorted(set(formal_core) - set(retained))),
        })
    return pd.DataFrame(rows)


def candidate_leave_one_out(patient_targets, unit_column):
    rows = []
    for unit in sorted(patient_targets[unit_column].unique()):
        subset = patient_targets[patient_targets[unit_column] != unit]
        for gene in FOCUS_TARGETS:
            gene_data = subset[subset["gene"] == gene]
            pvr = gene_data[gene_data["disease"] == "PVR"]["expressing_fraction"]
            pdr = gene_data[gene_data["disease"] == "PDR"]["expressing_fraction"]
            rows.append({
                f"excluded_{unit_column}": unit,
                "gene": gene,
                "remaining_patients": gene_data["sample"].nunique(),
                "remaining_PVR_patients": pvr.size,
                "remaining_PDR_patients": pdr.size,
                "PVR_median_expressing_fraction": float(pvr.median()) if len(pvr) else np.nan,
                "PDR_median_expressing_fraction": float(pdr.median()) if len(pdr) else np.nan,
                "all_remaining_patients_detected": bool((gene_data["expressing_cells"] > 0).all()),
            })
    return pd.DataFrame(rows)


def load_target_expression_for_cells(states, targets):
    frames = []
    for sample, sample_cells in states.groupby("sample", sort=False):
        genes = read_vector(MATRIX_ROOT / sample / "genes.tsv.gz")
        barcodes = read_vector(MATRIX_ROOT / sample / "barcodes.tsv.gz")
        gene_index = {gene: index for index, gene in enumerate(genes)}
        barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
        matrix = sparse.load_npz(MATRIX_ROOT / sample / "counts_gene_by_cell.npz").tocsr()
        cell_indices = [barcode_index[barcode] for barcode in sample_cells["barcode"]]
        totals = np.asarray(matrix[:, cell_indices].sum(axis=0)).ravel()
        target_counts = matrix[[gene_index[gene] for gene in targets]][:, cell_indices].toarray().T
        normalized = np.log1p(target_counts * np.divide(
            10000, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0,
        )[:, None])
        current = sample_cells.reset_index(drop=True).copy()
        for index, gene in enumerate(targets):
            current[gene] = normalized[:, index]
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def summarize_alternative_states(alternatives):
    rows = []
    official = alternatives["state_annotation"] == SHARED_STATE
    for name, current in alternatives.groupby("alternative_state"):
        selected = current["selected"]
        selected_cells = current[selected]
        intersection = int((selected & official.loc[current.index]).sum())
        rows.append({
            "alternative_state": name,
            "selected_cells": int(selected.sum()),
            "patients_covered": selected_cells["sample"].nunique(),
            "PVR_patients_covered": selected_cells.loc[selected_cells["disease"] == "PVR", "sample"].nunique(),
            "PDR_patients_covered": selected_cells.loc[selected_cells["disease"] == "PDR", "sample"].nunique(),
            "official_shared_cells_in_alternative": intersection,
            "official_shared_recall": intersection / int(official.loc[current.index].sum()),
            "alternative_precision_for_official_shared": intersection / int(selected.sum()),
            "mean_ECM_score": selected_cells["ECM_score"].mean(),
            "mean_TGFB_Mechanosensing_score": selected_cells["TGFB_Mechanosensing_score"].mean(),
        })
    return pd.DataFrame(rows)


def alternative_target_metrics(alternatives):
    rows = []
    for (name, disease, sample), group in alternatives[alternatives["selected"]].groupby(
        ["alternative_state", "disease", "sample"]
    ):
        for gene in ALL_CANDIDATES:
            values = group[gene].to_numpy(float)
            rows.append({
                "alternative_state": name,
                "disease": disease,
                "sample": sample,
                "gene": gene,
                "cells": len(values),
                "expressing_fraction": float((values > 0).mean()),
                "mean_log_expression": float(values.mean()),
            })
    patient = pd.DataFrame(rows)
    summary = patient.groupby(["alternative_state", "disease", "gene"], as_index=False).agg(
        patients=("sample", "nunique"),
        patients_any_expression=("expressing_fraction", lambda values: int((values > 0).sum())),
        median_patient_expressing_fraction=("expressing_fraction", "median"),
        median_patient_mean_log_expression=("mean_log_expression", "median"),
    )
    return patient, summary


def plot_threshold_grid(grid):
    pivot = grid.pivot(index="cpm_threshold", columns="required_patient_fraction", values="formal_core_retention_fraction")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="mako", ax=ax)
    ax.set_title("Formal 25-gene core retention across thresholds")
    ax.set_xlabel("Required patient fraction")
    ax.set_ylabel("CPM threshold")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_grid_core_retention.png", dpi=180)
    plt.close(fig)


def plot_leave_one_out(patient_core, cohort_core):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(range(len(patient_core)), patient_core["formal_core_retention_fraction"])
    axes[0].set_xticks(range(len(patient_core)), patient_core["excluded_sample"], rotation=90, fontsize=6)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Leave-one-patient-out core retention")
    axes[1].bar(cohort_core["excluded_accession"], cohort_core["formal_core_retention_fraction"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Leave-one-cohort-out core retention")
    fig.tight_layout()
    fig.savefig(OUT / "leave_one_out_core_retention.png", dpi=180)
    plt.close(fig)


def write_summary(grid, patient_core, cohort_core, alternative_summary, alternative_targets, patient_target, cohort_target):
    strict_grid = grid[(grid["cpm_threshold"] == 1) & (grid["required_patient_fraction"] == 1)].iloc[0]
    alternative_focus = alternative_targets[alternative_targets["gene"].isin(FOCUS_TARGETS)]
    min_alt_coverage = alternative_focus["patients_any_expression"].min()
    lines = [
        "# ",
        "",
        f"-  25  CPM >= 1  15/15 : {int(strict_grid['formal_core_retained'])}/25.",
        f"- : {patient_core['formal_core_retention_fraction'].min():.1%}.",
        f"- : {cohort_core['formal_core_retention_fraction'].min():.1%}.",
        f"-  {int(alternative_summary['patients_covered'].min())}/15 .",
        f"- `POSTN`, `CTHRC1`, `SULF1` : {int(min_alt_coverage)}.",
        "",
        "## ",
        "",
        grid.to_markdown(index=False),
        "",
        "## ",
        "",
        alternative_summary.to_markdown(index=False),
        "",
        "## ",
        "",
        "- 25 .",
        "-  top 20%/25%/30% ECM/TGF-beta , .",
        "- `POSTN`, `CTHRC1`, `SULF1` ;  PDR .",
        "-  GSE294329  1  PVR ;  GSE165784  PVR  PDR , .",
        "",
        "## ",
        "",
        "- , , .",
        "- , , .",
        "- , , .",
        "",
    ]
    (OUT / "robustness_sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(PSEUDO / "shared_state_pseudobulk_metadata.csv")
    cpm = load_cpm()
    markers = set(pd.read_csv(STATE / "shared_state_marker_genes.csv")["gene"])
    formal_core = list(pd.read_csv(PSEUDO / "robust_shared_state_genes.csv")["gene"])
    patient_targets = pd.read_csv(TARGET / "shared_state_target_expression_by_patient.csv")

    grid = threshold_grid(cpm, markers, [0.5, 1.0, 2.0], [0.8, 0.9, 1.0], formal_core)
    patient_core = leave_one_out_core(cpm, metadata, formal_core, "sample")
    cohort_core = leave_one_out_core(cpm, metadata, formal_core, "accession")
    patient_target = candidate_leave_one_out(patient_targets, "sample")
    cohort_target = candidate_leave_one_out(patient_targets, "accession")

    states = pd.read_csv(STATE / "fibrotic_cell_states.csv.gz")
    states = load_target_expression_for_cells(states, ALL_CANDIDATES)
    alternatives = define_alternative_states(states, [0.20, 0.25, 0.30])
    alternative_summary = summarize_alternative_states(alternatives)
    alternative_patient, alternative_targets = alternative_target_metrics(alternatives)

    grid.to_csv(OUT / "threshold_grid_shared_core.csv", index=False)
    patient_core.to_csv(OUT / "leave_one_patient_out_core.csv", index=False)
    cohort_core.to_csv(OUT / "leave_one_cohort_out_core.csv", index=False)
    patient_target.to_csv(OUT / "leave_one_patient_out_focus_targets.csv", index=False)
    cohort_target.to_csv(OUT / "leave_one_cohort_out_focus_targets.csv", index=False)
    alternative_summary.to_csv(OUT / "alternative_state_summary.csv", index=False)
    alternative_patient.to_csv(OUT / "alternative_state_target_expression_by_patient.csv", index=False)
    alternative_targets.to_csv(OUT / "alternative_state_target_summary.csv", index=False)
    plot_threshold_grid(grid)
    plot_leave_one_out(patient_core, cohort_core)
    write_summary(grid, patient_core, cohort_core, alternative_summary, alternative_targets, patient_target, cohort_target)


if __name__ == "__main__":
    main()
