from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ccs_signaling_analysis" / "results"
RESULTS = ROOT / "results"
SHARED_STATE = "Shared_ECM_TGFB_high"
RNG = np.random.default_rng(20260619)


PATHWAY_GENE_SETS = {
    "TGF_beta_SMAD": {
        "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "TGFBR3", "SMAD2", "SMAD3",
        "SMAD4", "SMAD7", "SERPINE1", "CTGF", "TGFBI", "THBS1", "THBS2", "INHBA",
    },
    "ECM_integrin_FAK": {
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "FN1", "POSTN", "TNC",
        "THBS1", "THBS2", "ITGA5", "ITGAV", "ITGB1", "ITGB3", "PTK2", "PXN",
        "TLN1", "VCL", "ACTN1",
    },
    "YAP_TAZ_TEAD": {
        "YAP1", "WWTR1", "TEAD1", "TEAD2", "TEAD3", "TEAD4", "CTGF", "CYR61",
        "ANKRD1", "AMOTL2", "AXL",
    },
    "NFkB_inflammatory": {
        "NFKB1", "RELA", "RELB", "NFKBIA", "NFKBIZ", "TNFAIP3", "ICAM1",
        "CCL2", "CXCL8", "IL6", "TNF", "IL1B",
    },
    "AP1_JUN_FOS": {"JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2", "ATF3", "EGR1"},
    "Hypoxia_HIF": {"HIF1A", "VEGFA", "ADM", "LDHA", "ENO1", "PDK1", "SLC2A1", "CA9", "BNIP3", "NDRG1"},
    "VEGF_axis": {"VEGFA", "VEGFB", "PGF", "KDR", "FLT1", "FLT4", "NRP1", "NRP2"},
    "PDGF_axis": {"PDGFA", "PDGFB", "PDGFC", "PDGFD", "PDGFRA", "PDGFRB"},
    "ECM_remodeling": {"MMP2", "MMP14", "TIMP1", "TIMP2", "LOX", "COL1A1", "COL1A2", "COL3A1", "FN1", "POSTN"},
}


LR_AXES = {
    "TGFB_TGFBR": {
        "ligands": ["TGFB1", "TGFB2", "TGFB3"],
        "receptors": ["TGFBR1", "TGFBR2", "TGFBR3"],
    },
    "POSTN_integrin": {
        "ligands": ["POSTN"],
        "receptors": ["ITGAV", "ITGB3", "ITGA5", "ITGB1", "ITGA6"],
    },
    "FN1_integrin": {
        "ligands": ["FN1"],
        "receptors": ["ITGA5", "ITGB1", "ITGAV", "ITGB3", "ITGA4"],
    },
    "THBS_TGFB_activation": {
        "ligands": ["THBS1", "THBS2"],
        "receptors": ["CD47", "ITGAV", "ITGB1", "TGFBR1", "TGFBR2"],
    },
    "PDGF_PDGFR": {
        "ligands": ["PDGFA", "PDGFB", "PDGFC", "PDGFD"],
        "receptors": ["PDGFRA", "PDGFRB"],
    },
    "VEGF_VEGFR": {
        "ligands": ["VEGFA", "VEGFB", "PGF"],
        "receptors": ["KDR", "FLT1", "FLT4", "NRP1", "NRP2"],
    },
    "SPP1_CD44_integrin": {
        "ligands": ["SPP1"],
        "receptors": ["CD44", "ITGAV", "ITGB1", "ITGB3"],
    },
    "CXCL_CXCR": {
        "ligands": ["CXCL1", "CXCL2", "CXCL3", "CXCL8", "CXCL12"],
        "receptors": ["CXCR1", "CXCR2", "CXCR4"],
    },
    "CCL_CCR": {
        "ligands": ["CCL2", "CCL3", "CCL4", "CCL5"],
        "receptors": ["CCR1", "CCR2", "CCR5"],
    },
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FIB = load_module(ROOT / "scripts" / "run_fibrotic_state_analysis.py", "fibrotic_state")


def read_vector(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


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
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    pd.DataFrame(rows).to_csv(output, index=False)


def bh_fdr(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    n = len(ranked)
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1] if n else np.array([])
    adjusted = np.minimum(adjusted, 1.0)
    temp = np.empty(n)
    temp[order] = adjusted
    out[valid] = temp
    return out


def bootstrap_mean_ci(values: np.ndarray, iterations: int = 3000) -> tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.nan, np.nan
    boot = [RNG.choice(values, len(values), replace=True).mean() for _ in range(iterations)]
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if len(left) == 0 or len(right) == 0:
        return np.nan
    return float(np.sign(left[:, None] - right[None, :]).mean())


def paired_effect_statistics(table: pd.DataFrame, metric: str, group_label: str) -> dict:
    paired = table.dropna(subset=[f"shared_{metric}", f"other_{metric}"]).copy()
    diff = paired[f"shared_{metric}"].to_numpy(float) - paired[f"other_{metric}"].to_numpy(float)
    if len(diff) and not np.allclose(diff, 0):
        wilcoxon_p = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    else:
        wilcoxon_p = np.nan
    low, high = bootstrap_mean_ci(diff)
    return {
        "feature": group_label,
        "patients": int(paired["sample"].nunique()),
        "positive_patients": int((diff > 0).sum()),
        "median_shared": float(paired[f"shared_{metric}"].median()) if len(paired) else np.nan,
        "median_other": float(paired[f"other_{metric}"].median()) if len(paired) else np.nan,
        "mean_difference_shared_minus_other": float(diff.mean()) if len(diff) else np.nan,
        "median_difference_shared_minus_other": float(np.median(diff)) if len(diff) else np.nan,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "wilcoxon_p": wilcoxon_p,
    }


def disease_statistics(table: pd.DataFrame, feature_column: str, value_column: str) -> pd.DataFrame:
    rows = []
    for feature, sub in table.groupby(feature_column):
        pdr = sub.loc[sub["disease"] == "PDR", value_column].to_numpy(float)
        pvr = sub.loc[sub["disease"] == "PVR", value_column].to_numpy(float)
        if len(pdr) and len(pvr):
            p = float(stats.mannwhitneyu(pdr, pvr, alternative="two-sided").pvalue)
        else:
            p = np.nan
        rows.append({
            feature_column: feature,
            "PDR_patients": len(pdr),
            "PVR_patients": len(pvr),
            "PDR_median": float(np.median(pdr)) if len(pdr) else np.nan,
            "PVR_median": float(np.median(pvr)) if len(pvr) else np.nan,
            "median_difference_PDR_minus_PVR": float(np.median(pdr) - np.median(pvr)) if len(pdr) and len(pvr) else np.nan,
            "cliffs_delta_PDR_vs_PVR": cliffs_delta(pdr, pvr),
            "mann_whitney_p": p,
        })
    result = pd.DataFrame(rows)
    result["mann_whitney_fdr"] = bh_fdr(result["mann_whitney_p"])
    return result


def attach_states(meta: pd.DataFrame) -> pd.DataFrame:
    states = pd.read_csv(RESULTS / "fibrotic_state_analysis" / "fibrotic_cell_states.csv.gz")
    labels = states.set_index(["sample", "barcode"])["state_annotation"]
    meta = meta.copy()
    meta["state_annotation"] = [labels.loc[(row.sample, row.barcode)] for row in meta.itertuples()]
    return meta


def score_gene_sets(lognorm: sparse.csr_matrix, genes: list[str], gene_sets: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = {gene: i for i, gene in enumerate(genes)}
    scores = {}
    coverage = []
    for name, gene_set in gene_sets.items():
        present = [gene for gene in sorted(gene_set) if gene in index]
        columns = [index[gene] for gene in present]
        if columns:
            scores[name] = np.asarray(lognorm[:, columns].mean(axis=1)).ravel()
        else:
            scores[name] = np.zeros(lognorm.shape[0])
        coverage.append({
            "feature": name,
            "requested_genes": len(gene_set),
            "present_genes": len(present),
            "present_gene_symbols": ";".join(present),
            "missing_gene_symbols": ";".join(sorted(set(gene_set) - set(present))),
        })
    return pd.DataFrame(scores), pd.DataFrame(coverage)


def pathway_activity_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts, genes, meta = FIB.load_target_cells()
    meta = attach_states(meta)
    lognorm = FIB.normalize_log(counts)
    scores, coverage = score_gene_sets(lognorm, genes, PATHWAY_GENE_SETS)
    coverage.to_csv(OUT / "pathway_gene_set_coverage.csv", index=False)

    cell_scores = pd.concat([meta[["accession", "disease", "sample", "barcode", "final_annotation", "state_annotation"]], scores], axis=1)
    state_summary = cell_scores.groupby(["accession", "disease", "sample", "state_annotation"], as_index=False).agg(
        cells=("barcode", "size"),
        **{name: (name, "median") for name in PATHWAY_GENE_SETS},
    )
    state_summary.to_csv(OUT / "pathway_activity_by_patient_state.csv", index=False)

    rows = []
    for sample, sub in cell_scores.groupby("sample"):
        disease = sub["disease"].iloc[0]
        accession = sub["accession"].iloc[0]
        shared = sub[sub["state_annotation"] == SHARED_STATE]
        other = sub[sub["state_annotation"] != SHARED_STATE]
        if shared.empty or other.empty:
            continue
        for pathway in PATHWAY_GENE_SETS:
            rows.append({
                "accession": accession,
                "disease": disease,
                "sample": sample,
                "pathway": pathway,
                "shared_score": float(shared[pathway].median()),
                "other_score": float(other[pathway].median()),
                "difference_shared_minus_other": float(shared[pathway].median() - other[pathway].median()),
                "shared_cells": len(shared),
                "other_cells": len(other),
            })
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT / "pathway_activity_shared_vs_other_by_patient.csv", index=False)

    stats_rows = []
    for pathway, sub in paired.groupby("pathway"):
        stats_rows.append(paired_effect_statistics(sub.rename(columns={"shared_score": "shared_score", "other_score": "other_score"}), "score", pathway))
    stats_table = pd.DataFrame(stats_rows)
    stats_table["wilcoxon_fdr"] = bh_fdr(stats_table["wilcoxon_p"])
    stats_table.to_csv(OUT / "pathway_activity_shared_vs_other_statistics.csv", index=False)

    disease_input = state_summary[state_summary["state_annotation"] == SHARED_STATE].melt(
        id_vars=["accession", "disease", "sample", "state_annotation", "cells"],
        value_vars=list(PATHWAY_GENE_SETS),
        var_name="pathway",
        value_name="score",
    )
    disease_stats = disease_statistics(disease_input, "pathway", "score")
    disease_stats.to_csv(OUT / "pathway_activity_shared_state_disease_statistics.csv", index=False)
    plot_pathway_effects(stats_table)
    return cell_scores, stats_table, disease_stats


def all_lr_genes() -> list[str]:
    genes = set()
    for axis in LR_AXES.values():
        genes.update(axis["ligands"])
        genes.update(axis["receptors"])
    genes.update({"POSTN", "FN1", "THBS1", "THBS2", "TIMP1", "SERPINE1", "CTHRC1"})
    return sorted(genes)


def load_all_cell_expression(selected_genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    metas = []
    coverage = []
    for accession, samples in FIB.COHORTS.items():
        annotations = pd.read_csv(RESULTS / "formal_clustering" / accession / "final_singlet_annotations.csv.gz")
        for sample in samples:
            sample_meta = annotations[annotations["sample"] == sample].copy()
            gene_list = read_vector(RESULTS / "formal_qc" / "filtered_matrices" / sample / "genes.tsv.gz")
            barcodes = read_vector(RESULTS / "formal_qc" / "filtered_matrices" / sample / "barcodes.tsv.gz")
            matrix = sparse.load_npz(RESULTS / "formal_qc" / "filtered_matrices" / sample / "counts_gene_by_cell.npz").tocsr()
            gene_index = {gene: i for i, gene in enumerate(gene_list)}
            barcode_index = {barcode: i for i, barcode in enumerate(barcodes)}
            present = [gene for gene in selected_genes if gene in gene_index]
            coverage.append({
                "sample": sample,
                "requested_genes": len(selected_genes),
                "present_genes": len(present),
                "missing_gene_symbols": ";".join(gene for gene in selected_genes if gene not in gene_index),
            })
            cell_indices = [barcode_index[barcode] for barcode in sample_meta["barcode"]]
            selected = np.zeros((len(cell_indices), len(selected_genes)), dtype=np.float32)
            if present:
                gene_rows = [gene_index[gene] for gene in present]
                raw = matrix[gene_rows][:, cell_indices].T.tocsr()
                totals = np.asarray(matrix[:, cell_indices].sum(axis=0)).ravel()
                raw = raw.multiply(np.divide(10000, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)[:, None])
                raw.data = np.log1p(raw.data)
                present_positions = [selected_genes.index(gene) for gene in present]
                selected[:, present_positions] = raw.toarray()
            rows.append(pd.DataFrame(selected, columns=selected_genes))
            meta = sample_meta[["sample", "barcode", "final_annotation"]].copy()
            meta.insert(0, "accession", accession)
            meta["disease"] = FIB.disease_for(sample)
            metas.append(meta)
    return pd.concat(metas, ignore_index=True), pd.concat(rows, ignore_index=True), pd.DataFrame(coverage)


def summarize_group_expression(meta: pd.DataFrame, expr: pd.DataFrame, group_cols: list[str], genes: list[str]) -> pd.DataFrame:
    rows = []
    for key, idx in meta.groupby(group_cols).groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        values = expr.iloc[list(idx)]
        cells = len(values)
        for gene in genes:
            vector = values[gene].to_numpy(float)
            rows.append({
                **base,
                "gene": gene,
                "cells": cells,
                "expressing_cells": int((vector > 0).sum()),
                "expressing_fraction": float((vector > 0).mean()) if cells else 0.0,
                "mean_log_expression": float(vector.mean()) if cells else 0.0,
                "median_log_expression": float(np.median(vector)) if cells else 0.0,
            })
    return pd.DataFrame(rows)


def build_summary_lookup(summary: pd.DataFrame, group_cols: list[str]) -> dict[tuple, dict]:
    lookup = {}
    for row in summary.itertuples(index=False):
        group_key = tuple(getattr(row, col) for col in group_cols)
        lookup[group_key + (row.gene,)] = {
            "cells": row.cells,
            "expressing_fraction": row.expressing_fraction,
            "mean_log_expression": row.mean_log_expression,
            "expressing_cells": row.expressing_cells,
        }
    return lookup


def curated_lr_axis_analysis(cell_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genes = all_lr_genes()
    all_meta, all_expr, gene_coverage = load_all_cell_expression(genes)
    gene_coverage.to_csv(OUT / "ligand_receptor_gene_coverage_by_sample.csv", index=False)

    sender_summary = summarize_group_expression(all_meta, all_expr, ["accession", "disease", "sample", "final_annotation"], genes)
    sender_summary.to_csv(OUT / "sender_ligand_expression_by_patient_celltype.csv", index=False)

    target_meta = cell_scores[["accession", "disease", "sample", "barcode", "state_annotation"]].copy()
    target_expr = all_expr.loc[all_meta.set_index(["sample", "barcode"]).index.isin(target_meta.set_index(["sample", "barcode"]).index)].copy()
    # Preserve target cell order by merging indices explicitly.
    all_key = pd.Series(range(len(all_meta)), index=pd.MultiIndex.from_frame(all_meta[["sample", "barcode"]]))
    target_indices = [all_key.loc[(row.sample, row.barcode)] for row in target_meta.itertuples()]
    target_expr = all_expr.iloc[target_indices].reset_index(drop=True)
    receiver_summary = summarize_group_expression(target_meta.reset_index(drop=True), target_expr, ["accession", "disease", "sample", "state_annotation"], genes)
    receiver_summary.to_csv(OUT / "receiver_receptor_expression_by_patient_state.csv", index=False)

    sender_lookup = build_summary_lookup(sender_summary, ["accession", "disease", "sample", "final_annotation"])
    receiver_lookup = build_summary_lookup(receiver_summary, ["accession", "disease", "sample", "state_annotation"])
    samples = sorted(target_meta["sample"].unique())
    sender_groups = all_meta.groupby("sample")["final_annotation"].unique().to_dict()
    receiver_states = target_meta.groupby("sample")["state_annotation"].unique().to_dict()

    axis_rows = []
    pair_rows = []
    for sample in samples:
        disease = target_meta.loc[target_meta["sample"] == sample, "disease"].iloc[0]
        accession = target_meta.loc[target_meta["sample"] == sample, "accession"].iloc[0]
        for sender in sorted(sender_groups[sample]):
            for receiver_state in sorted(receiver_states[sample]):
                for axis_name, axis in LR_AXES.items():
                    scores = []
                    detected = 0
                    pair_count = 0
                    for ligand in axis["ligands"]:
                        ligand_info = sender_lookup.get((accession, disease, sample, sender, ligand))
                        if ligand_info is None:
                            continue
                        for receptor in axis["receptors"]:
                            receptor_info = receiver_lookup.get((accession, disease, sample, receiver_state, receptor))
                            if receptor_info is None:
                                continue
                            pair_count += 1
                            pair_score = ligand_info["mean_log_expression"] * receptor_info["mean_log_expression"]
                            pair_detected = (
                                ligand_info["cells"] >= 10
                                and receptor_info["cells"] >= 10
                                and ligand_info["expressing_fraction"] >= 0.05
                                and receptor_info["expressing_fraction"] >= 0.05
                            )
                            detected += int(pair_detected)
                            scores.append(pair_score)
                            pair_rows.append({
                                "accession": accession,
                                "disease": disease,
                                "sample": sample,
                                "axis": axis_name,
                                "sender_annotation": sender,
                                "receiver_state": receiver_state,
                                "ligand": ligand,
                                "receptor": receptor,
                                "sender_cells": ligand_info["cells"],
                                "receiver_cells": receptor_info["cells"],
                                "ligand_expressing_fraction": ligand_info["expressing_fraction"],
                                "receptor_expressing_fraction": receptor_info["expressing_fraction"],
                                "ligand_mean_log_expression": ligand_info["mean_log_expression"],
                                "receptor_mean_log_expression": receptor_info["mean_log_expression"],
                                "pair_score": pair_score,
                                "pair_detected": pair_detected,
                            })
                    axis_rows.append({
                        "accession": accession,
                        "disease": disease,
                        "sample": sample,
                        "axis": axis_name,
                        "sender_annotation": sender,
                        "receiver_state": receiver_state,
                        "pair_count": pair_count,
                        "detected_pair_count": detected,
                        "axis_detected": detected > 0,
                        "axis_score_mean": float(np.mean(scores)) if scores else 0.0,
                        "axis_score_max": float(np.max(scores)) if scores else 0.0,
                    })

    pair_table = pd.DataFrame(pair_rows)
    axis_table = pd.DataFrame(axis_rows)
    pair_table.to_csv(OUT / "curated_ligand_receptor_pair_scores.csv", index=False)
    axis_table.to_csv(OUT / "curated_ligand_receptor_axis_by_patient_sender_receiver.csv", index=False)

    best_rows = []
    for (sample, axis_name), sub in axis_table.groupby(["sample", "axis"]):
        shared = sub[sub["receiver_state"] == SHARED_STATE]
        other = sub[sub["receiver_state"] != SHARED_STATE]
        accession = sub["accession"].iloc[0]
        disease = sub["disease"].iloc[0]
        shared_best = shared.sort_values("axis_score_max", ascending=False).head(1)
        other_best = other.sort_values("axis_score_max", ascending=False).head(1)
        best_rows.append({
            "accession": accession,
            "disease": disease,
            "sample": sample,
            "axis": axis_name,
            "shared_axis_score": float(shared_best["axis_score_max"].iloc[0]) if len(shared_best) else np.nan,
            "other_axis_score": float(other_best["axis_score_max"].iloc[0]) if len(other_best) else np.nan,
            "difference_shared_minus_other": (
                float(shared_best["axis_score_max"].iloc[0] - other_best["axis_score_max"].iloc[0])
                if len(shared_best) and len(other_best) else np.nan
            ),
            "shared_best_sender": shared_best["sender_annotation"].iloc[0] if len(shared_best) else "",
            "other_best_sender": other_best["sender_annotation"].iloc[0] if len(other_best) else "",
            "shared_detected_pair_count": int(shared_best["detected_pair_count"].iloc[0]) if len(shared_best) else 0,
            "other_detected_pair_count": int(other_best["detected_pair_count"].iloc[0]) if len(other_best) else 0,
        })
    best = pd.DataFrame(best_rows)
    best.to_csv(OUT / "curated_ligand_receptor_axis_patient_best_scores.csv", index=False)

    stat_rows = []
    for axis_name, sub in best.groupby("axis"):
        renamed = sub.rename(columns={"shared_axis_score": "shared_score", "other_axis_score": "other_score"})
        stat_rows.append(paired_effect_statistics(renamed, "score", axis_name))
    stats_table = pd.DataFrame(stat_rows)
    stats_table["wilcoxon_fdr"] = bh_fdr(stats_table["wilcoxon_p"])
    stats_table.to_csv(OUT / "curated_ligand_receptor_axis_shared_vs_other_statistics.csv", index=False)

    top_sender = axis_table[axis_table["receiver_state"] == SHARED_STATE].groupby(["axis", "sender_annotation"], as_index=False).agg(
        patients=("sample", "nunique"),
        detected_patients=("axis_detected", "sum"),
        median_shared_axis_score=("axis_score_max", "median"),
        median_detected_pair_count=("detected_pair_count", "median"),
    )
    top_sender["sender_rank_within_axis"] = top_sender.groupby("axis")["median_shared_axis_score"].rank(ascending=False, method="first")
    top_sender.sort_values(["axis", "sender_rank_within_axis"]).to_csv(OUT / "curated_ligand_receptor_top_sender_summary.csv", index=False)
    plot_axis_effects(stats_table)
    plot_top_sender_heatmap(top_sender)
    return axis_table, best, stats_table


def postn_node_summary(cell_scores: pd.DataFrame, lr_best: pd.DataFrame) -> pd.DataFrame:
    genes = ["POSTN", "ITGAV", "ITGB3", "ITGA5", "ITGB1", "ITGA6", "FN1", "THBS1", "THBS2", "TIMP1", "SERPINE1", "CTHRC1"]
    all_meta, all_expr, _ = load_all_cell_expression(genes)
    target_meta = cell_scores[["accession", "disease", "sample", "barcode", "state_annotation"]].copy()
    all_key = pd.Series(range(len(all_meta)), index=pd.MultiIndex.from_frame(all_meta[["sample", "barcode"]]))
    target_indices = [all_key.loc[(row.sample, row.barcode)] for row in target_meta.itertuples()]
    target_expr = all_expr.iloc[target_indices].reset_index(drop=True)
    target_meta = target_meta.reset_index(drop=True)
    shared = target_meta["state_annotation"].eq(SHARED_STATE)
    rows = []
    for sample, idx in target_meta[shared].groupby("sample").groups.items():
        sub_expr = target_expr.iloc[list(idx)]
        sub_meta = target_meta.iloc[list(idx)]
        disease = sub_meta["disease"].iloc[0]
        accession = sub_meta["accession"].iloc[0]
        best_postn = lr_best[(lr_best["sample"] == sample) & (lr_best["axis"] == "POSTN_integrin")]
        best_fn1 = lr_best[(lr_best["sample"] == sample) & (lr_best["axis"] == "FN1_integrin")]
        best_thbs = lr_best[(lr_best["sample"] == sample) & (lr_best["axis"] == "THBS_TGFB_activation")]
        postn = sub_expr["POSTN"].to_numpy(float)
        integrin_score = sub_expr[[g for g in ["ITGAV", "ITGB3", "ITGA5", "ITGB1", "ITGA6"] if g in sub_expr]].mean(axis=1)
        rows.append({
            "accession": accession,
            "disease": disease,
            "sample": sample,
            "shared_state_cells": len(sub_expr),
            "POSTN_expressing_fraction": float((postn > 0).mean()),
            "POSTN_mean_log_expression": float(postn.mean()),
            "integrin_receptor_mean_score": float(integrin_score.mean()),
            "POSTN_integrin_best_axis_score": float(best_postn["shared_axis_score"].iloc[0]) if len(best_postn) else np.nan,
            "POSTN_integrin_best_sender": best_postn["shared_best_sender"].iloc[0] if len(best_postn) else "",
            "FN1_integrin_best_axis_score": float(best_fn1["shared_axis_score"].iloc[0]) if len(best_fn1) else np.nan,
            "THBS_TGFB_activation_best_axis_score": float(best_thbs["shared_axis_score"].iloc[0]) if len(best_thbs) else np.nan,
            "TIMP1_mean_log_expression": float(sub_expr["TIMP1"].mean()),
            "SERPINE1_mean_log_expression": float(sub_expr["SERPINE1"].mean()),
            "CTHRC1_mean_log_expression": float(sub_expr["CTHRC1"].mean()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "postn_ecm_interaction_node_by_patient.csv", index=False)

    summary = {
        "patients": len(result),
        "POSTN_detected_patients": int((result["POSTN_expressing_fraction"] > 0).sum()),
        "POSTN_fraction_ge_10pct_patients": int((result["POSTN_expressing_fraction"] >= 0.10).sum()),
        "median_POSTN_expressing_fraction": float(result["POSTN_expressing_fraction"].median()),
        "median_integrin_receptor_score": float(result["integrin_receptor_mean_score"].median()),
        "median_POSTN_integrin_axis_score": float(result["POSTN_integrin_best_axis_score"].median()),
        "most_common_POSTN_integrin_sender": result["POSTN_integrin_best_sender"].mode().iloc[0] if len(result) else "",
    }
    pd.DataFrame([summary]).to_csv(OUT / "postn_ecm_interaction_node_summary.csv", index=False)
    plot_postn_node(result)
    return result


def plot_pathway_effects(stats_table: pd.DataFrame) -> None:
    plot = stats_table.sort_values("mean_difference_shared_minus_other")
    fig, ax = plt.subplots(figsize=(8, 5.8))
    y = np.arange(len(plot))
    diff = plot["mean_difference_shared_minus_other"].to_numpy(float)
    low = plot["bootstrap_ci_low"].to_numpy(float)
    high = plot["bootstrap_ci_high"].to_numpy(float)
    ax.barh(y, diff, color="#4c78a8", alpha=0.85)
    ax.errorbar(diff, y, xerr=[diff - low, high - diff], fmt="none", ecolor="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, plot["feature"])
    ax.set_xlabel("Patient-paired median score difference\nShared state minus other fibrotic states")
    ax.set_title("Curated signaling-pathway activity in the shared state")
    fig.tight_layout()
    fig.savefig(OUT / "pathway_activity_shared_vs_other_effects.png", dpi=220)
    plt.close(fig)


def plot_axis_effects(stats_table: pd.DataFrame) -> None:
    plot = stats_table.sort_values("mean_difference_shared_minus_other")
    fig, ax = plt.subplots(figsize=(8, 5.8))
    y = np.arange(len(plot))
    diff = plot["mean_difference_shared_minus_other"].to_numpy(float)
    low = plot["bootstrap_ci_low"].to_numpy(float)
    high = plot["bootstrap_ci_high"].to_numpy(float)
    ax.barh(y, diff, color="#f58518", alpha=0.85)
    ax.errorbar(diff, y, xerr=[diff - low, high - diff], fmt="none", ecolor="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, plot["feature"])
    ax.set_xlabel("Best patient-level axis score difference\nShared state minus other fibrotic states")
    ax.set_title("Curated ligand-receptor axis evidence")
    fig.tight_layout()
    fig.savefig(OUT / "curated_ligand_receptor_axis_shared_vs_other_effects.png", dpi=220)
    plt.close(fig)


def plot_top_sender_heatmap(top_sender: pd.DataFrame) -> None:
    pivot = top_sender.pivot(index="axis", columns="sender_annotation", values="median_shared_axis_score").fillna(0.0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.75), 5.5))
    im = ax.imshow(pivot.to_numpy(float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
    ax.set_title("Top sender-cell evidence for curated axes into the shared state")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median axis score")
    fig.tight_layout()
    fig.savefig(OUT / "curated_ligand_receptor_top_sender_heatmap.png", dpi=220)
    plt.close(fig)


def plot_postn_node(result: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("POSTN_expressing_fraction", "POSTN+ fraction"),
        ("integrin_receptor_mean_score", "Shared-state integrin score"),
        ("POSTN_integrin_best_axis_score", "POSTN-integrin axis score"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for disease, color, x in [("PVR", "#d95f5f", 0), ("PDR", "#4c78a8", 1)]:
            vals = result.loc[result["disease"] == disease, metric].to_numpy(float)
            jitter = RNG.normal(0, 0.04, len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, color=color, alpha=0.85)
            if len(vals):
                ax.hlines(np.median(vals), x - 0.22, x + 0.22, color="black", lw=2)
        ax.set_xticks([0, 1], ["PVR", "PDR"])
        ax.set_title(title)
    fig.suptitle("POSTN as an ECM-cell interaction node, not a proven therapeutic mechanism")
    fig.tight_layout()
    fig.savefig(OUT / "postn_ecm_interaction_node.png", dpi=220)
    plt.close(fig)


def plot_mechanism_model() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc, ec="#333333", size=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.15)

    def arrow(x1, y1, x2, y2, label="", dashed=False):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333", linestyle="--" if dashed else "-"),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.035, label, ha="center", va="center", fontsize=8)

    box(
        0.04, 0.64, 0.25, 0.17,
        "PVR-enriched context\nRPE-like inflammatory program\nMuller stress program",
        "#fde2e2",
    )
    box(
        0.04, 0.31, 0.25, 0.18,
        "PDR-enriched context\nPericyte / vascular-wall\ncontractile program\nEndothelial/angiogenic context",
        "#dbeafe",
        size=9,
    )
    box(
        0.38, 0.47, 0.25, 0.23,
        "Convergent shared state\nShared_ECM_TGFB_high\n2,335 cells; 15/15 patients\nECM + TGF-beta\n+ mechanosensing",
        "#fef3c7",
        size=10,
    )
    box(
        0.71, 0.57, 0.25, 0.22,
        "Curated signaling-axis context\nTGFB-TGFBR\nPOSTN/FN1-integrin\nTHBS-TGF-beta activation\nPDGF/VEGF; SPP1; CXCL/CCL",
        "#e9d5ff",
        size=9,
    )
    box(
        0.71, 0.27, 0.25, 0.20,
        "ECM-cell interaction nodes\nPOSTN, FN1, THBS1/2\nTIMP1, SERPINE1, CTHRC1\nCandidate biology\nnot causal proof",
        "#dcfce7",
        size=9,
    )
    box(
        0.39, 0.10, 0.22, 0.14,
        "Fibrotic membrane remodeling\nmatrix deposition\ntraction/contraction\nnon-VEGF prioritization",
        "#e5e7eb",
        size=9,
    )

    arrow(0.29, 0.72, 0.38, 0.61, "source-associated inputs")
    arrow(0.29, 0.40, 0.38, 0.56, "source-associated inputs")
    arrow(0.63, 0.61, 0.71, 0.68, "axis context")
    arrow(0.63, 0.51, 0.71, 0.37, "ECM nodes")
    arrow(0.84, 0.57, 0.52, 0.24, "", dashed=True)
    arrow(0.51, 0.47, 0.51, 0.24, "state-level convergence")
    ax.text(
        0.50,
        0.03,
        "Interpretation boundary: scRNA-seq supports patient-level signaling-state convergence and ligand/receptor expression context; "
        "it does not prove spatial proximity, lineage conversion, or treatment efficacy.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(OUT / "figure5_ccs_mechanism_model_draft.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(pathway_stats: pd.DataFrame, pathway_disease: pd.DataFrame, axis_stats: pd.DataFrame, postn: pd.DataFrame) -> None:
    positive_pathways = pathway_stats[pathway_stats["positive_patients"] >= 12].sort_values("mean_difference_shared_minus_other", ascending=False)
    positive_axes = axis_stats[axis_stats["positive_patients"] >= 12].sort_values("mean_difference_shared_minus_other", ascending=False)
    postn_summary = pd.read_csv(OUT / "postn_ecm_interaction_node_summary.csv").iloc[0]
    lines = [
        "# CCS ",
        "",
        "## ",
        "",
        " Cell Communication and Signaling , , ."
        " signaling-axis evidence, , .",
        "",
        "## ",
        "",
        f"- , {len(positive_pathways)}/{len(pathway_stats)}  12/15 .",
        f"-  ligand-receptor , {len(positive_axes)}/{len(axis_stats)}  12/15 .",
        f"- POSTN  {int(postn_summary['POSTN_detected_patients'])}/{int(postn_summary['patients'])} , "
        f" {int(postn_summary['POSTN_fraction_ge_10pct_patients'])}/{int(postn_summary['patients'])}  10%.",
        f"- POSTN-integrin  `{postn_summary['most_common_POSTN_integrin_sender']}`; "
        " POSTN  ECM-cell interaction node, .",
        "",
        "## ",
        "",
        "1.  Figure 5 :  ->  ->  ECM/TGF-beta/mechanotransduction  -> ECM  -> .",
        "2. : Curated signaling-axis analysis supports an ECM-integrin/TGF-beta communication context of the shared state.",
        "3.  POSTN  leading target  matrix-cell interaction node with the strongest multi-layer support.",
        "",
        "## ",
        "",
        "-  CellChat/; .",
        "-  POSTN-integrin ; , .",
        "-  PDGF/VEGF/CXCL/CCL .",
        "",
        "## ",
        "",
        pathway_stats.sort_values("mean_difference_shared_minus_other", ascending=False).to_markdown(index=False),
        "",
        "##  ligand-receptor ",
        "",
        axis_stats.sort_values("mean_difference_shared_minus_other", ascending=False).to_markdown(index=False),
        "",
        "##  PDR/PVR , ",
        "",
        pathway_disease.sort_values("median_difference_PDR_minus_PVR", ascending=False).to_markdown(index=False),
    ]
    (OUT / "ccs_signaling_axis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_paths = [
        RESULTS / "fibrotic_state_analysis" / "fibrotic_cell_states.csv.gz",
        RESULTS / "fibrotic_state_analysis" / "state_cluster_summary.csv",
        RESULTS / "formal_clustering" / "sample_annotation_counts.csv",
        ROOT / "scripts" / "run_fibrotic_state_analysis.py",
    ]
    write_manifest(input_paths, OUT / "input_manifest.csv")
    cell_scores, pathway_stats, pathway_disease = pathway_activity_analysis()
    _axis_table, lr_best, axis_stats = curated_lr_axis_analysis(cell_scores)
    postn = postn_node_summary(cell_scores, lr_best)
    plot_mechanism_model()
    write_summary(pathway_stats, pathway_disease, axis_stats, postn)
    output_paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_manifest(output_paths, OUT / "output_manifest.csv")
    print(f"CCS signaling-axis analysis complete. Outputs: {OUT}")


if __name__ == "__main__":
    main()
