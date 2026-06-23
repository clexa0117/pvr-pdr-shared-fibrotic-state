import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = ROOT / "results/formal_qc/filtered_matrices"
STATE_ROOT = ROOT / "results/fibrotic_state_analysis"
OUT = ROOT / "results/patient_pseudobulk"

PROGRAM_COLUMNS = [
    "ECM_median",
    "Contractile_median",
    "TGFB_Mechanosensing_median",
    "RPE_origin_median",
    "Glial_origin_median",
    "Pericyte_origin_median",
]


def finite_fdr(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    valid = np.isfinite(pvalues)
    adjusted[valid] = multipletests(pvalues[valid], method="fdr_bh")[1]
    return adjusted


def read_vector(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def permutation_pvalue(values_a, values_b, iterations=20000, seed=0):
    rng = np.random.default_rng(seed)
    observed = abs(np.mean(values_a) - np.mean(values_b))
    combined = np.concatenate([values_a, values_b])
    count = 1
    for _ in range(iterations):
        shuffled = rng.permutation(combined)
        difference = abs(np.mean(shuffled[:len(values_a)]) - np.mean(shuffled[len(values_a):]))
        count += difference >= observed
    return count / (iterations + 1)


def cliffs_delta(values_a, values_b):
    comparisons = np.sign(values_a[:, None] - values_b[None, :])
    return float(comparisons.mean())


def patient_level_statistics(table, metrics):
    rows = []
    pvr = table[table["disease"] == "PVR"]
    pdr = table[table["disease"] == "PDR"]
    for metric in metrics:
        a = pdr[metric].to_numpy(float)
        b = pvr[metric].to_numpy(float)
        mann = stats.mannwhitneyu(a, b, alternative="two-sided")
        welch = stats.ttest_ind(a, b, equal_var=False)
        rows.append({
            "metric": metric,
            "PDR_patients": len(a),
            "PVR_patients": len(b),
            "PDR_median": float(np.median(a)),
            "PVR_median": float(np.median(b)),
            "median_difference_PDR_minus_PVR": float(np.median(a) - np.median(b)),
            "cliffs_delta_PDR_vs_PVR": cliffs_delta(a, b),
            "mann_whitney_p": float(mann.pvalue),
            "welch_t_p": float(welch.pvalue),
            "permutation_p": permutation_pvalue(a, b),
        })
    result = pd.DataFrame(rows)
    for column in ["mann_whitney_p", "welch_t_p", "permutation_p"]:
        result[column.replace("_p", "_fdr")] = finite_fdr(result[column])
    return result


def cohort_sensitivity_statistics(table, metrics):
    rows = []
    g294_pvr = table[table["accession"] == "GSE294329"]
    g245_pdr = table[table["accession"] == "GSE245561"]
    g165_pdr = table[(table["accession"] == "GSE165784") & (table["disease"] == "PDR")]
    g165_pvr = table[(table["accession"] == "GSE165784") & (table["disease"] == "PVR")]
    for metric in metrics:
        effect_245 = float(g245_pdr[metric].median() - g294_pvr[metric].median())
        effect_165 = float(g165_pdr[metric].median() - g294_pvr[metric].median())
        rows.append({
            "metric": metric,
            "GSE294329_PVR_median": float(g294_pvr[metric].median()),
            "GSE245561_PDR_median": float(g245_pdr[metric].median()),
            "GSE165784_PDR_median": float(g165_pdr[metric].median()),
            "GSE165784_auxiliary_PVR_value": float(g165_pvr[metric].iloc[0]) if len(g165_pvr) else np.nan,
            "effect_GSE245561_PDR_minus_GSE294329_PVR": effect_245,
            "effect_GSE165784_PDR_minus_GSE294329_PVR": effect_165,
            "same_direction_both_PDR_cohorts": bool(np.sign(effect_245) == np.sign(effect_165)),
        })
    return pd.DataFrame(rows)


def build_shared_pseudobulk(states):
    shared = states[states["state_annotation"] == "Shared_ECM_TGFB_high"]
    samples = sorted(shared["sample"].unique())
    gene_lists = {sample: read_vector(MATRIX_ROOT / sample / "genes.tsv.gz") for sample in samples}
    common = set(gene_lists[samples[0]])
    for genes in gene_lists.values():
        common &= set(genes)
    genes = sorted(common)

    rows = []
    metadata = []
    for sample in samples:
        sample_cells = shared[shared["sample"] == sample]
        matrix = sparse.load_npz(MATRIX_ROOT / sample / "counts_gene_by_cell.npz").tocsr()
        barcodes = read_vector(MATRIX_ROOT / sample / "barcodes.tsv.gz")
        barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
        gene_index = {gene: index for index, gene in enumerate(gene_lists[sample])}
        cell_indices = [barcode_index[barcode] for barcode in sample_cells["barcode"]]
        selected = matrix[[gene_index[gene] for gene in genes]][:, cell_indices]
        rows.append(np.asarray(selected.sum(axis=1)).ravel())
        first = sample_cells.iloc[0]
        metadata.append({
            "sample": sample,
            "accession": first["accession"],
            "disease": first["disease"],
            "shared_state_cells": len(sample_cells),
        })
    counts = np.vstack(rows).astype(np.int64)
    metadata = pd.DataFrame(metadata)
    return counts, genes, metadata


def pseudobulk_gene_statistics(counts, genes, metadata):
    library = counts.sum(axis=1)
    cpm = counts / library[:, None] * 1e6
    logcpm = np.log2(cpm + 1)
    pvr = metadata["disease"].to_numpy() == "PVR"
    pdr = ~pvr
    pvalues = np.ones(len(genes))
    mann_p = np.ones(len(genes))
    for index in range(len(genes)):
        pvalues[index] = stats.ttest_ind(logcpm[pdr, index], logcpm[pvr, index], equal_var=False).pvalue
        mann_p[index] = stats.mannwhitneyu(logcpm[pdr, index], logcpm[pvr, index], alternative="two-sided").pvalue
    overall_effect = np.mean(logcpm[pdr], axis=0) - np.mean(logcpm[pvr], axis=0)

    g294_pvr = metadata["accession"].to_numpy() == "GSE294329"
    g245_pdr = metadata["accession"].to_numpy() == "GSE245561"
    g165_pdr = (metadata["accession"].to_numpy() == "GSE165784") & pdr
    effect_245_vs_294 = np.mean(logcpm[g245_pdr], axis=0) - np.mean(logcpm[g294_pvr], axis=0)
    effect_165pdr_vs_294 = np.mean(logcpm[g165_pdr], axis=0) - np.mean(logcpm[g294_pvr], axis=0)
    same_direction = (
        (np.sign(effect_245_vs_294) == np.sign(effect_165pdr_vs_294))
        & (np.sign(effect_245_vs_294) == np.sign(overall_effect))
    )

    detected_patients = (cpm >= 1).sum(axis=0)
    result = pd.DataFrame({
        "gene": genes,
        "detected_patients_cpm_ge_1": detected_patients,
        "median_cpm": np.median(cpm, axis=0),
        "mean_log2cpm_PDR": np.mean(logcpm[pdr], axis=0),
        "mean_log2cpm_PVR": np.mean(logcpm[pvr], axis=0),
        "log2cpm_effect_PDR_minus_PVR": overall_effect,
        "welch_p": pvalues,
        "mann_whitney_p": mann_p,
        "effect_GSE245561_PDR_vs_GSE294329_PVR": effect_245_vs_294,
        "effect_GSE165784_PDR_vs_GSE294329_PVR": effect_165pdr_vs_294,
        "same_direction_both_PDR_cohorts": same_direction,
    })
    result["welch_fdr"] = finite_fdr(result["welch_p"])
    result["mann_whitney_fdr"] = finite_fdr(result["mann_whitney_p"])
    return result, cpm, logcpm


def robust_shared_genes(gene_stats, shared_markers):
    marker_set = set(shared_markers["gene"])
    result = gene_stats[
        (gene_stats["gene"].isin(marker_set))
        & (gene_stats["detected_patients_cpm_ge_1"] == 15)
    ].copy()
    return result.sort_values(["median_cpm"], ascending=False)


def plot_patient_programs(shared_summary):
    long = shared_summary.melt(
        id_vars=["accession", "disease", "sample"],
        value_vars=["ECM_median", "Contractile_median", "TGFB_Mechanosensing_median"],
        var_name="program", value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=long, x="program", y="score", hue="disease", ax=ax, showfliers=False)
    sns.stripplot(data=long, x="program", y="score", hue="disease", dodge=True, ax=ax, color="black", size=4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False)
    ax.set_title("Patient-level program scores in Shared_ECM_TGFB_high")
    fig.tight_layout()
    fig.savefig(OUT / "shared_state_patient_programs.png", dpi=180)
    plt.close(fig)


def plot_abundance(abundance):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=abundance, x="disease", y="shared_state_fraction_of_target", ax=ax, showfliers=False)
    sns.stripplot(data=abundance, x="disease", y="shared_state_fraction_of_target", ax=ax, color="black", size=5)
    ax.set_title("Shared-state fraction among fibrosis-related target cells")
    fig.tight_layout()
    fig.savefig(OUT / "shared_state_patient_abundance.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(STATE_ROOT / "fibrotic_cell_states.csv.gz")
    shared_summary = pd.read_csv(STATE_ROOT / "shared_state_patient_summary.csv")

    total_target = states.groupby(["accession", "disease", "sample"], as_index=False).size().rename(columns={"size": "target_cells"})
    shared_count = states[states["state_annotation"] == "Shared_ECM_TGFB_high"].groupby(
        ["accession", "disease", "sample"], as_index=False
    ).size().rename(columns={"size": "shared_state_cells"})
    abundance = total_target.merge(shared_count, on=["accession", "disease", "sample"], how="left").fillna({"shared_state_cells": 0})
    abundance["shared_state_fraction_of_target"] = abundance["shared_state_cells"] / abundance["target_cells"]
    abundance.to_csv(OUT / "patient_shared_state_abundance.csv", index=False)

    program_stats = patient_level_statistics(shared_summary, PROGRAM_COLUMNS)
    program_stats.to_csv(OUT / "shared_state_program_statistics.csv", index=False)
    program_sensitivity = cohort_sensitivity_statistics(shared_summary, PROGRAM_COLUMNS)
    program_sensitivity.to_csv(OUT / "shared_state_program_cohort_sensitivity.csv", index=False)
    abundance_stats = patient_level_statistics(abundance, ["shared_state_fraction_of_target"])
    abundance_stats.to_csv(OUT / "shared_state_abundance_statistics.csv", index=False)
    abundance_sensitivity = cohort_sensitivity_statistics(abundance, ["shared_state_fraction_of_target"])
    abundance_sensitivity.to_csv(OUT / "shared_state_abundance_cohort_sensitivity.csv", index=False)

    counts, genes, metadata = build_shared_pseudobulk(states)
    gene_stats, cpm, logcpm = pseudobulk_gene_statistics(counts, genes, metadata)
    gene_stats.to_csv(OUT / "shared_state_gene_statistics.csv.gz", index=False)
    pd.DataFrame(counts, index=metadata["sample"], columns=genes).to_csv(OUT / "shared_state_pseudobulk_counts.csv.gz")
    pd.DataFrame(logcpm, index=metadata["sample"], columns=genes).to_csv(OUT / "shared_state_pseudobulk_log2cpm.csv.gz")
    metadata.to_csv(OUT / "shared_state_pseudobulk_metadata.csv", index=False)

    markers = pd.read_csv(STATE_ROOT / "shared_state_marker_genes.csv")
    robust_shared_genes(gene_stats, markers).to_csv(OUT / "robust_shared_state_genes.csv", index=False)
    exploratory = gene_stats[
        gene_stats["same_direction_both_PDR_cohorts"]
        & (gene_stats["detected_patients_cpm_ge_1"] >= 12)
        & (gene_stats["welch_fdr"] < 0.10)
        & (gene_stats["log2cpm_effect_PDR_minus_PVR"].abs() >= 1)
    ].sort_values("welch_fdr")
    exploratory.to_csv(OUT / "exploratory_disease_gene_differences.csv", index=False)
    pd.DataFrame(columns=[
        "gene", "reason_not_accepted",
    ]).to_csv(OUT / "formal_disease_specific_genes.csv", index=False)

    plot_patient_programs(shared_summary)
    plot_abundance(abundance)

    robust = pd.read_csv(OUT / "robust_shared_state_genes.csv")
    lines = [
        "#  pseudobulk ", "",
        f"-  pseudobulk: {len(metadata)} , PVR 6 , PDR 9 ",
        f"- : {len(robust)} ( 15/15  CPM >= 1)",
        f"- : {len(exploratory)} (FDR < 0.10, || >= 1,  PDR )",
        "- : 0 (, )",
        "",
        "## ", "",
        "- , .",
        "-  GEO ,  PVR/PDR .",
        "- , .",
        "- /, .",
        "- GSE165784  PVR  ECM  PDR,  PVR/PDR .",
        "",
        "## ", "",
        program_stats.to_markdown(index=False),
        "",
        "## ", "",
        abundance_stats.to_markdown(index=False),
        "",
        "## ", "",
        " PDR  GSE294329 PVR ; GSE165784  PVR .",
        "",
        program_sensitivity.to_markdown(index=False),
        "",
        abundance_sensitivity.to_markdown(index=False),
        "",
        "## ", "",
        "- : 25  15/15  CPM >= 1, .",
        "- : TGF-beta/ PVR  PDR , .",
        "- : PDR  ECM, ,  PVR .",
        "- :  PVR/PDR ; .",
    ]
    (OUT / "patient_pseudobulk_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
