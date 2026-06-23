from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
OUT = ANALYSIS_DIR / "results"
SHARED_STATE = "Shared_ECM_TGFB_high"
RNG = np.random.default_rng(20260619)


TF_TARGET_GENE_SETS = {
    "SMAD2_3_TGFbeta_targets": {
        "SERPINE1",
        "SMAD7",
        "CTGF",
        "TGFBI",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "THBS1",
        "THBS2",
        "INHBA",
        "MMP2",
    },
    "TEAD_YAP_TAZ_targets": {
        "CTGF",
        "CYR61",
        "ANKRD1",
        "AMOTL2",
        "AXL",
        "THBS1",
        "FSTL3",
        "MYC",
        "BIRC5",
    },
    "RELA_NFKB_targets": {
        "NFKBIA",
        "TNFAIP3",
        "ICAM1",
        "VCAM1",
        "CCL2",
        "CXCL1",
        "CXCL2",
        "CXCL8",
        "IL6",
        "PTGS2",
    },
    "JUN_FOS_AP1_targets": {
        "JUN",
        "JUNB",
        "JUND",
        "FOS",
        "FOSB",
        "FOSL1",
        "FOSL2",
        "ATF3",
        "EGR1",
        "MMP1",
        "MMP3",
        "MMP9",
    },
}


RECEPTOR_GENE_SETS = {
    "Integrin_matrix_receptors": {"ITGA5", "ITGAV", "ITGB1", "ITGB3", "ITGA6", "ITGA4", "CD44"},
    "TGF_beta_receptors": {"TGFBR1", "TGFBR2", "TGFBR3", "ACVR1", "ACVR2A", "ACVR2B"},
    "POSTN_FN1_binding_receptors": {"ITGA5", "ITGAV", "ITGB1", "ITGB3", "ITGA6"},
    "SPP1_CD44_integrin_receptors": {"CD44", "ITGAV", "ITGB1", "ITGB3"},
    "PDGF_receptors": {"PDGFRA", "PDGFRB"},
    "VEGF_receptors": {"KDR", "FLT1", "FLT4", "NRP1", "NRP2"},
    "Chemokine_receptors": {"CXCR1", "CXCR2", "CXCR4", "CCR1", "CCR2", "CCR5"},
}


FOCAL_RECEPTOR_GENES = sorted(set().union(*RECEPTOR_GENE_SETS.values()))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_module(ANALYSIS_DIR / "scripts" / "run_ccs_signaling_axis_analysis.py", "ccs_axis")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(paths: list[Path], output: Path) -> None:
    rows = []
    for path in paths:
        if path.exists():
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def load_target_cell_matrix():
    counts, genes, meta = BASE.FIB.load_target_cells()
    meta = BASE.attach_states(meta)
    lognorm = BASE.FIB.normalize_log(counts)
    return genes, meta, lognorm


def paired_state_statistics(cell_scores: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for sample, sub in cell_scores.groupby("sample"):
        shared = sub[sub["state_annotation"] == SHARED_STATE]
        other = sub[sub["state_annotation"] != SHARED_STATE]
        if shared.empty or other.empty:
            continue
        disease = sub["disease"].iloc[0]
        accession = sub["accession"].iloc[0]
        for feature in features:
            rows.append(
                {
                    "accession": accession,
                    "disease": disease,
                    "sample": sample,
                    "feature": feature,
                    "shared_cells": len(shared),
                    "other_cells": len(other),
                    "shared_score": float(shared[feature].median()),
                    "other_score": float(other[feature].median()),
                    "difference_shared_minus_other": float(shared[feature].median() - other[feature].median()),
                }
            )
    paired = pd.DataFrame(rows)
    stat_rows = []
    for feature, sub in paired.groupby("feature"):
        stat_rows.append(BASE.paired_effect_statistics(sub, "score", feature))
    stats_table = pd.DataFrame(stat_rows)
    stats_table["wilcoxon_fdr"] = BASE.bh_fdr(stats_table["wilcoxon_p"])
    stats_table["manuscript_use"] = [
        classify_feature(row)
        for row in stats_table.itertuples(index=False)
    ]
    disease = BASE.disease_statistics(paired.rename(columns={"feature": "feature_name"}), "feature_name", "shared_score")
    disease = disease.rename(columns={"feature_name": "feature"})
    return paired, stats_table, disease


def classify_feature(row) -> str:
    mean_diff = getattr(row, "mean_difference_shared_minus_other")
    fdr = getattr(row, "wilcoxon_fdr")
    positive = getattr(row, "positive_patients")
    if np.isfinite(fdr) and fdr < 0.05 and mean_diff > 0 and positive >= 12:
        return "supports_shared_state_context"
    if np.isfinite(fdr) and fdr < 0.05 and mean_diff < 0:
        return "lower_in_shared_state_do_not_emphasize"
    return "exploratory_or_not_supported"


def score_gene_set_panel(genes: list[str], meta: pd.DataFrame, lognorm, gene_sets: dict[str, set[str]], prefix: str):
    scores, coverage = BASE.score_gene_sets(lognorm, genes, gene_sets)
    coverage.to_csv(OUT / f"{prefix}_gene_set_coverage.csv", index=False)
    cell_scores = pd.concat(
        [
            meta[["accession", "disease", "sample", "barcode", "final_annotation", "state_annotation"]].reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    state_summary = cell_scores.groupby(["accession", "disease", "sample", "state_annotation"], as_index=False).agg(
        cells=("barcode", "size"),
        **{name: (name, "median") for name in gene_sets},
    )
    state_summary.to_csv(OUT / f"{prefix}_activity_by_patient_state.csv", index=False)
    paired, stats_table, disease = paired_state_statistics(cell_scores, list(gene_sets))
    paired.to_csv(OUT / f"{prefix}_activity_shared_vs_other_by_patient.csv", index=False)
    stats_table.to_csv(OUT / f"{prefix}_activity_shared_vs_other_statistics.csv", index=False)
    disease.to_csv(OUT / f"{prefix}_activity_shared_state_disease_statistics.csv", index=False)
    return cell_scores, paired, stats_table, disease


def receptor_gene_expression_analysis(genes: list[str], meta: pd.DataFrame, lognorm) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_index = {gene: i for i, gene in enumerate(genes)}
    present = [gene for gene in FOCAL_RECEPTOR_GENES if gene in gene_index]
    expr = pd.DataFrame({gene: np.asarray(lognorm[:, gene_index[gene]].todense()).ravel() for gene in present})
    cell_expr = pd.concat(
        [
            meta[["accession", "disease", "sample", "barcode", "final_annotation", "state_annotation"]].reset_index(drop=True),
            expr.reset_index(drop=True),
        ],
        axis=1,
    )
    state_rows = []
    for (accession, disease, sample, state), sub in cell_expr.groupby(["accession", "disease", "sample", "state_annotation"]):
        for gene in present:
            vals = sub[gene].to_numpy(float)
            state_rows.append(
                {
                    "accession": accession,
                    "disease": disease,
                    "sample": sample,
                    "state_annotation": state,
                    "gene": gene,
                    "cells": len(sub),
                    "mean_log_expression": float(vals.mean()),
                    "median_log_expression": float(np.median(vals)),
                    "expressing_fraction": float((vals > 0).mean()),
                }
            )
    state_table = pd.DataFrame(state_rows)
    state_table.to_csv(OUT / "receptor_gene_availability_by_patient_state.csv", index=False)

    paired_rows = []
    for sample, sub in cell_expr.groupby("sample"):
        shared = sub[sub["state_annotation"] == SHARED_STATE]
        other = sub[sub["state_annotation"] != SHARED_STATE]
        if shared.empty or other.empty:
            continue
        for gene in present:
            sv = shared[gene].to_numpy(float)
            ov = other[gene].to_numpy(float)
            paired_rows.append(
                {
                    "accession": sub["accession"].iloc[0],
                    "disease": sub["disease"].iloc[0],
                    "sample": sample,
                    "gene": gene,
                    "shared_cells": len(shared),
                    "other_cells": len(other),
                    "shared_mean_log_expression": float(sv.mean()),
                    "other_mean_log_expression": float(ov.mean()),
                    "difference_mean_log_expression": float(sv.mean() - ov.mean()),
                    "shared_expressing_fraction": float((sv > 0).mean()),
                    "other_expressing_fraction": float((ov > 0).mean()),
                    "difference_expressing_fraction": float((sv > 0).mean() - (ov > 0).mean()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "receptor_gene_shared_vs_other_by_patient.csv", index=False)

    stats_rows = []
    for gene, sub in paired.groupby("gene"):
        temp = sub.rename(
            columns={
                "shared_mean_log_expression": "shared_score",
                "other_mean_log_expression": "other_score",
            }
        )
        stat = BASE.paired_effect_statistics(temp, "score", gene)
        stat["metric"] = "mean_log_expression"
        stats_rows.append(stat)
        temp = sub.rename(
            columns={
                "shared_expressing_fraction": "shared_score",
                "other_expressing_fraction": "other_score",
            }
        )
        stat = BASE.paired_effect_statistics(temp, "score", gene)
        stat["metric"] = "expressing_fraction"
        stats_rows.append(stat)
    stats_table = pd.DataFrame(stats_rows)
    stats_table["wilcoxon_fdr"] = stats_table.groupby("metric", group_keys=False)["wilcoxon_p"].transform(
        lambda x: pd.Series(BASE.bh_fdr(x), index=x.index)
    )
    stats_table["manuscript_use"] = [
        classify_feature(row)
        for row in stats_table.itertuples(index=False)
    ]
    stats_table.to_csv(OUT / "receptor_gene_shared_vs_other_statistics.csv", index=False)
    return state_table, stats_table


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#464C55",
            "ytick.color": "#464C55",
            "grid.color": "#E6E8F0",
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
        }
    )


def plot_effects(table: pd.DataFrame, output: Path, title: str, subtitle: str, color: str) -> None:
    set_plot_style()
    plot = table.sort_values("mean_difference_shared_minus_other")
    fig, ax = plt.subplots(figsize=(8.7, max(4.2, 0.42 * len(plot) + 1.4)))
    y = np.arange(len(plot))
    diff = plot["mean_difference_shared_minus_other"].to_numpy(float)
    low = plot["bootstrap_ci_low"].to_numpy(float)
    high = plot["bootstrap_ci_high"].to_numpy(float)
    ax.barh(y, diff, color=color, edgecolor="#464C55", linewidth=0.7, alpha=0.9)
    ax.errorbar(diff, y, xerr=[diff - low, high - diff], fmt="none", ecolor="#1F2430", lw=1)
    ax.axvline(0, color="#1F2430", lw=1)
    labels = [
        f"{row.feature} ({int(row.positive_patients)}/{int(row.patients)})"
        for row in plot.itertuples()
    ]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Patient-paired median score difference: shared state minus other states")
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=13, color="#1F2430", pad=20)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="#6F768A")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_receptor_gene_effects(stats_table: pd.DataFrame) -> None:
    set_plot_style()
    plot = stats_table[stats_table["metric"] == "mean_log_expression"].copy()
    plot = plot.sort_values("mean_difference_shared_minus_other").tail(18)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    y = np.arange(len(plot))
    diff = plot["mean_difference_shared_minus_other"].to_numpy(float)
    low = plot["bootstrap_ci_low"].to_numpy(float)
    high = plot["bootstrap_ci_high"].to_numpy(float)
    colors = ["#A3BEFA" if value >= 0 else "#E2E5EA" for value in diff]
    ax.barh(y, diff, color=colors, edgecolor="#464C55", linewidth=0.7)
    ax.errorbar(diff, y, xerr=[diff - low, high - diff], fmt="none", ecolor="#1F2430", lw=1)
    ax.axvline(0, color="#1F2430", lw=1)
    ax.set_yticks(y, plot["feature"])
    ax.set_xlabel("Mean log-expression difference: shared state minus other states")
    ax.set_title("Focal receptor-gene availability in the shared state", loc="left", fontsize=13, pad=20)
    ax.text(
        0,
        1.02,
        "Top receptor genes by positive patient-paired mean-expression difference; labels retain patient-level uncertainty.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#6F768A",
    )
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(OUT / "receptor_gene_mean_expression_effects.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_summary(tf_stats: pd.DataFrame, receptor_stats: pd.DataFrame, receptor_gene_stats: pd.DataFrame) -> None:
    tf_supported = tf_stats[tf_stats["manuscript_use"] == "supports_shared_state_context"].sort_values(
        "mean_difference_shared_minus_other", ascending=False
    )
    receptor_supported = receptor_stats[receptor_stats["manuscript_use"] == "supports_shared_state_context"].sort_values(
        "mean_difference_shared_minus_other", ascending=False
    )
    receptor_gene_supported = receptor_gene_stats[
        (receptor_gene_stats["metric"] == "mean_log_expression")
        & (receptor_gene_stats["manuscript_use"] == "supports_shared_state_context")
    ].sort_values("mean_difference_shared_minus_other", ascending=False)

    lines = [
        "# CCS TF and receptor-availability supplement",
        "",
        "## Analysis role",
        "",
        "This supplement tests whether the shared fibrotic state carries an inferred regulatory and receptor-availability context suitable for a Cell Communication and Signaling submission. The analysis remains expression based and patient paired; it does not infer causal transcription-factor activity, spatial communication, or receptor activation.",
        "",
        "## Main findings",
        "",
        f"- TF target-program screen: {len(tf_supported)}/{len(tf_stats)} target programs were higher in the shared state in at least 12/15 patients with FDR < 0.05.",
        f"- Receptor panel screen: {len(receptor_supported)}/{len(receptor_stats)} receptor panels were higher in the shared state in at least 12/15 patients with FDR < 0.05.",
        f"- Focal receptor-gene screen: {len(receptor_gene_supported)} receptor genes met the same patient-level support rule for mean log expression.",
        "",
        "## Manuscript-use recommendation",
        "",
        "- If TF target-program support is dominated by SMAD/TEAD/NF-kB-related programs, it can be added as one sentence to the existing curated signaling-axis result.",
        "- Receptor availability should be used as context for ECM-cell interaction competence, not as evidence that a ligand-receptor interaction occurred.",
        "- Weak or lower-in-shared-state receptor panels should remain in supplementary results or the exploratory boundary statement.",
        "",
        "## TF target-program statistics",
        "",
        tf_stats.sort_values("mean_difference_shared_minus_other", ascending=False).to_markdown(index=False),
        "",
        "## Receptor-panel statistics",
        "",
        receptor_stats.sort_values("mean_difference_shared_minus_other", ascending=False).to_markdown(index=False),
        "",
        "## Focal receptor-gene statistics, mean log expression",
        "",
        receptor_gene_stats[receptor_gene_stats["metric"] == "mean_log_expression"]
        .sort_values("mean_difference_shared_minus_other", ascending=False)
        .to_markdown(index=False),
    ]
    (OUT / "ccs_tf_receptor_availability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_paths = [
        ROOT / "results" / "fibrotic_state_analysis" / "fibrotic_cell_states.csv.gz",
        ROOT / "results" / "formal_clustering" / "sample_annotation_counts.csv",
        ROOT / "scripts" / "run_fibrotic_state_analysis.py",
        ANALYSIS_DIR / "scripts" / "run_ccs_signaling_axis_analysis.py",
    ]
    write_manifest(input_paths, OUT / "input_manifest_tf_receptor.csv")

    genes, meta, lognorm = load_target_cell_matrix()
    _tf_cell, _tf_paired, tf_stats, _tf_disease = score_gene_set_panel(
        genes, meta, lognorm, TF_TARGET_GENE_SETS, "tf_target_program"
    )
    _receptor_cell, _receptor_paired, receptor_stats, _receptor_disease = score_gene_set_panel(
        genes, meta, lognorm, RECEPTOR_GENE_SETS, "receptor_availability"
    )
    _receptor_state, receptor_gene_stats = receptor_gene_expression_analysis(genes, meta, lognorm)

    plot_effects(
        tf_stats,
        OUT / "tf_target_program_shared_vs_other_effects.png",
        "Inferred TF target-program context of the shared state",
        "Patient-paired comparisons; values in parentheses show positive patients.",
        "#A3BEFA",
    )
    plot_effects(
        receptor_stats,
        OUT / "receptor_availability_shared_vs_other_effects.png",
        "Receptor-availability context of the shared state",
        "Expression context only; not receptor activation or physical communication.",
        "#F0986E",
    )
    plot_receptor_gene_effects(receptor_gene_stats)
    write_summary(tf_stats, receptor_stats, receptor_gene_stats)

    output_paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_manifest(output_paths, OUT / "output_manifest.csv")
    print(f"CCS TF/receptor availability analysis complete. Outputs: {OUT}")


if __name__ == "__main__":
    main()
