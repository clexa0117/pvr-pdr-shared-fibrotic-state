import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = ROOT / "results/formal_qc/filtered_matrices"
ANNOTATION_ROOT = ROOT / "results/formal_clustering"
STATE_PATH = ROOT / "results/fibrotic_state_analysis/fibrotic_cell_states.csv.gz"
PRIORITY_PATH = ROOT / "results/external_validation_targets/prioritized_non_vegf_targets.csv"
RISK_PATH = ROOT / "results/external_validation_targets/target_risk_review.csv"
OUT = ROOT / "results/target_cellular_specificity"

TARGETS = ["POSTN", "FN1", "THBS2", "CTHRC1", "AEBP1", "IGFBP7", "SULF1", "CFH", "TIMP1", "SERPINE1"]
TARGET_TYPES = {"Fibroblast_Mesenchymal", "RPE_like", "Muller_Glial", "Pericyte"}
SHARED_STATE = "Shared_ECM_TGFB_high"
EXPERIMENTAL_ROLES = {
    "POSTN": "Primary intervention target",
    "CTHRC1": "Cross-disease mechanistic comparator",
    "THBS2": "PDR-weighted state-specific comparator",
    "AEBP1": "PDR-weighted mechanistic comparator",
    "FN1": "Broad ECM positive control; safety-limited target",
    "IGFBP7": "Broad endothelial-associated emerging marker",
    "SULF1": "Cross-disease emerging comparator",
    "CFH": "Retinal-protective non-inhibition safety control",
    "TIMP1": "Context-dependent risk control",
    "SERPINE1": "Context-dependent risk control",
}


def read_vector(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def disease_for(sample):
    return "PVR" if sample.startswith("GSM890") or "RRD-ERM1" in sample else "PDR"


def summarize_expression(cells, group_columns, genes):
    rows = []
    grouped = cells.groupby(group_columns, observed=True, dropna=False)
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for gene in genes:
            values = group[gene].to_numpy(dtype=float)
            expressing = values > 0
            rows.append({
                **base,
                "gene": gene,
                "cells": len(values),
                "expressing_cells": int(expressing.sum()),
                "expressing_fraction": float(expressing.mean()),
                "mean_log_expression": float(values.mean()),
                "mean_log_expression_among_expressing": float(values[expressing].mean()) if expressing.any() else 0.0,
            })
    return pd.DataFrame(rows)


def shared_state_specificity(cells, state_column, shared_state, genes):
    shared = cells[cells[state_column] == shared_state]
    other = cells[cells[state_column] != shared_state]
    rows = []
    for gene in genes:
        shared_values = shared[gene].to_numpy(dtype=float)
        other_values = other[gene].to_numpy(dtype=float)
        shared_fraction = float((shared_values > 0).mean())
        other_fraction = float((other_values > 0).mean())
        rows.append({
            "gene": gene,
            "shared_cells": len(shared_values),
            "other_target_state_cells": len(other_values),
            "shared_expressing_fraction": shared_fraction,
            "other_expressing_fraction": other_fraction,
            "expressing_fraction_difference": shared_fraction - other_fraction,
            "expressing_fraction_ratio": shared_fraction / other_fraction if other_fraction > 0 else np.inf,
            "shared_mean_log_expression": float(shared_values.mean()),
            "other_mean_log_expression": float(other_values.mean()),
            "mean_log_expression_difference": float(shared_values.mean() - other_values.mean()),
        })
    return pd.DataFrame(rows)


def load_all_cells():
    metadata_parts = []
    expression_parts = []
    for annotation_path in sorted(ANNOTATION_ROOT.glob("*/final_singlet_annotations.csv.gz")):
        accession = annotation_path.parent.name
        annotations = pd.read_csv(annotation_path)
        for sample, selected in annotations.groupby("sample", sort=False):
            genes = read_vector(MATRIX_ROOT / sample / "genes.tsv.gz")
            barcodes = read_vector(MATRIX_ROOT / sample / "barcodes.tsv.gz")
            matrix = sparse.load_npz(MATRIX_ROOT / sample / "counts_gene_by_cell.npz").tocsr()
            gene_index = {gene: index for index, gene in enumerate(genes)}
            barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
            missing = sorted(set(TARGETS) - set(gene_index))
            if missing:
                raise ValueError(f"{sample} is missing target genes: {missing}")
            cell_indices = [barcode_index[barcode] for barcode in selected["barcode"]]
            totals = np.asarray(matrix[:, cell_indices].sum(axis=0)).ravel()
            target_counts = matrix[[gene_index[gene] for gene in TARGETS]][:, cell_indices].toarray().T
            normalized = np.log1p(target_counts * np.divide(
                10000,
                totals,
                out=np.zeros_like(totals, dtype=float),
                where=totals > 0,
            )[:, None])
            meta = selected[["sample", "barcode", "final_annotation"]].reset_index(drop=True)
            meta.insert(0, "accession", accession)
            meta["disease"] = disease_for(sample)
            metadata_parts.append(meta)
            expression_parts.append(pd.DataFrame(normalized, columns=TARGETS))
    metadata = pd.concat(metadata_parts, ignore_index=True)
    expression = pd.concat(expression_parts, ignore_index=True)
    return pd.concat([metadata, expression], axis=1)


def add_state_annotations(all_cells):
    states = pd.read_csv(STATE_PATH, usecols=["sample", "barcode", "state_annotation"])
    cells = all_cells.merge(states, on=["sample", "barcode"], how="left", validate="one_to_one")
    cells["state_annotation"] = cells["state_annotation"].fillna("Not_fibrosis_target")
    return cells


def patient_shared_metrics(cells):
    shared = cells[cells["state_annotation"] == SHARED_STATE]
    patient = summarize_expression(shared, ["accession", "disease", "sample"], TARGETS)
    coverage = patient.groupby("gene", as_index=False).agg(
        patients=("sample", "nunique"),
        patients_any_expression=("expressing_cells", lambda values: int((values > 0).sum())),
        patients_expression_fraction_ge_10pct=("expressing_fraction", lambda values: int((values >= 0.10).sum())),
        median_patient_expressing_fraction=("expressing_fraction", "median"),
        min_patient_expressing_fraction=("expressing_fraction", "min"),
        median_patient_mean_log_expression=("mean_log_expression", "median"),
    )
    disease_coverage = patient.groupby(["gene", "disease"], as_index=False).agg(
        disease_patients=("sample", "nunique"),
        disease_patients_any_expression=("expressing_cells", lambda values: int((values > 0).sum())),
        median_expressing_fraction=("expressing_fraction", "median"),
    )
    return patient, coverage, disease_coverage


def add_disease_balance(priority, disease_coverage):
    balance = disease_coverage.pivot(index="gene", columns="disease", values="median_expressing_fraction").reset_index()
    balance = balance.rename(columns={
        "PVR": "PVR_median_patient_expressing_fraction",
        "PDR": "PDR_median_patient_expressing_fraction",
    })
    pvr = balance["PVR_median_patient_expressing_fraction"]
    pdr = balance["PDR_median_patient_expressing_fraction"]
    balance["cross_disease_balance_ratio"] = np.minimum(pvr, pdr) / np.maximum(pvr, pdr).replace(0, np.nan)
    return priority.merge(balance, on="gene", how="left")


def build_experimental_priority(cells, specificity, coverage, disease_coverage):
    tiers = pd.concat([
        pd.read_csv(PRIORITY_PATH)[["gene", "translational_tier"]],
        pd.read_csv(RISK_PATH)[["gene", "translational_tier"]],
    ], ignore_index=True)
    non_target = cells[~cells["final_annotation"].isin(TARGET_TYPES)]
    off_target = summarize_expression(non_target, ["final_annotation"], TARGETS)
    maximum = off_target.sort_values(["gene", "expressing_fraction"], ascending=[True, False]).groupby("gene", as_index=False).first()
    maximum = maximum[["gene", "final_annotation", "expressing_fraction", "mean_log_expression"]].rename(columns={
        "final_annotation": "highest_non_target_cell_type",
        "expressing_fraction": "highest_non_target_expressing_fraction",
        "mean_log_expression": "highest_non_target_mean_log_expression",
    })
    result = tiers.merge(specificity, on="gene", how="left").merge(coverage, on="gene", how="left").merge(maximum, on="gene", how="left")
    result = add_disease_balance(result, disease_coverage)
    result["experimental_role"] = result["gene"].map(EXPERIMENTAL_ROLES)
    return result


def dotplot(summary, group_column, filename, title, group_order=None):
    plot = summary.copy()
    if group_order is None:
        group_order = list(dict.fromkeys(plot[group_column]))
    y_index = {value: index for index, value in enumerate(group_order)}
    x_index = {gene: index for index, gene in enumerate(TARGETS)}
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.55 * len(group_order))))
    points = ax.scatter(
        plot["gene"].map(x_index),
        plot[group_column].map(y_index),
        s=20 + 650 * plot["expressing_fraction"],
        c=plot["mean_log_expression"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_xticks(range(len(TARGETS)), TARGETS, rotation=45, ha="right")
    ax.set_yticks(range(len(group_order)), group_order)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Candidate target")
    ax.set_ylabel("")
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Mean log-normalized expression")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def patient_heatmap(patient):
    matrix = patient.pivot(index="sample", columns="gene", values="expressing_fraction").reindex(columns=TARGETS)
    diseases = patient.drop_duplicates("sample").set_index("sample")["disease"]
    matrix = matrix.loc[sorted(matrix.index, key=lambda sample: (diseases[sample], sample))]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(matrix, cmap="mako", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Expressing-cell fraction"})
    ax.set_title("Candidate-target expression across patients in Shared_ECM_TGFB_high")
    ax.set_xlabel("")
    ax.set_ylabel("Patient sample")
    fig.tight_layout()
    fig.savefig(OUT / "shared_state_patient_target_heatmap.png", dpi=180)
    plt.close(fig)


def write_summary(cells, specificity, coverage, priority):
    shared = cells[cells["state_annotation"] == SHARED_STATE]
    top_specific = specificity.sort_values("expressing_fraction_difference", ascending=False).head(5)["gene"].tolist()
    broad_non_target = priority.sort_values("highest_non_target_expressing_fraction", ascending=False).head(3)["gene"].tolist()
    lines = [
        "# ",
        "",
        f"- : {len(cells):,}",
        f"- : {int((cells['state_annotation'] != 'Not_fibrosis_target').sum()):,}",
        f"- : {len(shared):,}",
        f"- : {len(TARGETS)}",
        "",
        "## ",
        "",
        f"- : `{'`, `'.join(top_specific)}`.",
    ]
    for gene in TARGETS:
        row = coverage[coverage["gene"] == gene].iloc[0]
        lines.append(
            f"- `{gene}`:  {int(row['patients_any_expression'])}/{int(row['patients'])} , "
            f"{int(row['patients_expression_fraction_ge_10pct'])}/{int(row['patients'])}  10%, "
            f" {row['median_patient_expressing_fraction']:.1%}."
        )
    lines += [
        "",
        "## ",
        "",
    ]
    for gene in TARGETS:
        row = priority[priority["gene"] == gene].iloc[0]
        lines.append(
            f"- `{gene}`: {row['experimental_role']};  `{row['translational_tier']}`; "
            f" {row['shared_expressing_fraction']:.1%},  {row['other_expressing_fraction']:.1%}; "
            f"PVR/PDR  {row['PVR_median_patient_expressing_fraction']:.1%}/{row['PDR_median_patient_expressing_fraction']:.1%}; "
            f" `{row['highest_non_target_cell_type']}`({row['highest_non_target_expressing_fraction']:.1%})."
        )
    lines += [
        "",
        "## ",
        "",
        "- `POSTN` : , ,  PDR,  PVR  PDR .",
        "- `SULF1` : PVR/PDR , , .",
        "- `CTHRC1` : ,  PDR , .",
        "- `THBS2`  `AEBP1`  PDR , .",
        "- `FN1`, `IGFBP7`  `TIMP1` ,  ECM/.",
        "",
        "## ",
        "",
        "- , .",
        "- , .",
        "- , .",
        "- ,  PDR; .",
        f"- : `{'`, `'.join(broad_non_target)}`.",
        "",
    ]
    (OUT / "target_cellular_specificity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_validation_plan(priority):
    lookup = priority.set_index("gene")
    lines = [
        "# POSTN ",
        "",
        "## : ",
        "",
        "-  PVR  PDR  POSTN ,  COL1A1/ACTA2 .",
        "-  CTHRC1  SULF1, ; FN1  ECM .",
        "-  THBS2  AEBP1  PDR , .",
        "- , .",
        "",
        "## : ",
        "",
        "-  ECM/TGF-beta  RPE ,  PVR  PDR .",
        "-  TGF-beta /,  POSTN, CTHRC1, THBS2, FN1, COL1A1, ACTA2 .",
        "-  POSTN , CTHRC1 , SULF1 ;  VEGF .",
        "",
        "## : ",
        "",
        "- : , , ECM , COL1A1/ACTA2/POSTN .",
        "- : RPE , .",
        "- `CFH` ; `TIMP1`  `SERPINE1` , .",
        "",
        "## ",
        "",
    ]
    for gene in ["POSTN", "SULF1", "CTHRC1", "THBS2", "AEBP1", "FN1"]:
        row = lookup.loc[gene]
        lines.append(
            f"- `{gene}`:  {row['shared_expressing_fraction']:.1%}, "
            f"15  {int(row['patients_any_expression'])} ; "
            f"PVR/PDR  {row['PVR_median_patient_expressing_fraction']:.1%}/{row['PDR_median_patient_expressing_fraction']:.1%}; "
            f" `{row['highest_non_target_cell_type']}`."
        )
    (OUT / "experimental_validation_plan.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = add_state_annotations(load_all_cells())
    cell_type = summarize_expression(cells, ["disease", "final_annotation"], TARGETS)
    fibrotic = cells[cells["state_annotation"] != "Not_fibrosis_target"].copy()
    state = summarize_expression(fibrotic, ["disease", "state_annotation"], TARGETS)
    specificity = shared_state_specificity(fibrotic, "state_annotation", SHARED_STATE, TARGETS)
    patient, coverage, disease_coverage = patient_shared_metrics(cells)
    priority = build_experimental_priority(cells, specificity, coverage, disease_coverage)

    cell_type.to_csv(OUT / "target_expression_by_cell_type_disease.csv", index=False)
    state.to_csv(OUT / "target_expression_by_fibrotic_state_disease.csv", index=False)
    specificity.to_csv(OUT / "shared_state_target_specificity.csv", index=False)
    patient.to_csv(OUT / "shared_state_target_expression_by_patient.csv", index=False)
    coverage.to_csv(OUT / "shared_state_target_patient_coverage.csv", index=False)
    disease_coverage.to_csv(OUT / "shared_state_target_patient_coverage_by_disease.csv", index=False)
    priority.to_csv(OUT / "target_experimental_priority.csv", index=False)

    cell_plot = cell_type.copy()
    cell_plot["disease_cell_type"] = cell_plot["disease"] + " | " + cell_plot["final_annotation"]
    dotplot(cell_plot, "disease_cell_type", "target_cell_type_dotplot.png", "Candidate targets across major cell types")
    state_plot = state.copy()
    state_plot["disease_state"] = state_plot["disease"] + " | " + state_plot["state_annotation"]
    dotplot(state_plot, "disease_state", "target_fibrotic_state_dotplot.png", "Candidate targets across fibrotic states")
    patient_heatmap(patient)
    write_summary(cells, specificity, coverage, priority)
    write_validation_plan(priority)


if __name__ == "__main__":
    main()
