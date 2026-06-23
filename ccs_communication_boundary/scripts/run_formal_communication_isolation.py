from __future__ import annotations

import gzip
import hashlib
import importlib.util
import math
import shutil
import subprocess
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ANALYSIS_DIR / "results"
RESOURCES = ANALYSIS_DIR / "resources"
SHARED_STATE = "Shared_ECM_TGFB_high"
RNG = np.random.default_rng(20260622)

OMNIPATH_URL = (
    "https://omnipathdb.org/interactions"
    "?datasets=ligrecextra&organisms=9606&genesymbols=true"
    "&fields=sources,references,curation_effort&format=tsv"
)

CORE_TARGETS_PATH = ROOT / "results" / "patient_pseudobulk" / "robust_shared_state_genes.csv"
MARKERS_PATH = ROOT / "results" / "fibrotic_state_analysis" / "shared_state_marker_genes.csv"

PYTHON_PACKAGES = ["liana", "cellphonedb", "scanpy", "anndata", "omnipath", "decoupler"]
R_PACKAGES = ["CellChat", "nichenetr", "OmnipathR", "Seurat", "SingleCellExperiment"]
STATE_PATH = ROOT / "results" / "fibrotic_state_analysis" / "fibrotic_cell_states.csv.gz"


def find_script(script_name: str, required_parent_hint: str | None = None) -> Path:
    matches = sorted(ROOT.glob(f"*/scripts/{script_name}"))
    if required_parent_hint is not None:
        matches = [path for path in matches if required_parent_hint.lower() in str(path).lower()]
    if not matches:
        raise FileNotFoundError(f"Could not locate {script_name}")
    return matches[0]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FIB = load_module(ROOT / "scripts" / "run_fibrotic_state_analysis.py", "fibrotic_state")
BASE = load_module(find_script("run_ccs_signaling_axis_analysis.py"), "ccs_axis")
TFMOD = load_module(find_script("run_ccs_tf_receptor_availability_analysis.py"), "ccs_tf")


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
    for path in sorted(set(paths)):
        if path.exists():
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def package_feasibility_table() -> pd.DataFrame:
    rows = []
    for package in PYTHON_PACKAGES:
        rows.append(
            {
                "language": "Python",
                "package": package,
                "available": importlib.util.find_spec(package) is not None,
                "use_in_this_isolation": "not_used_native_package",
            }
        )
    rscript = shutil.which("Rscript")
    for package in R_PACKAGES:
        if not rscript:
            available = False
            status = "Rscript_not_found"
        else:
            cmd = [
                rscript,
                "--vanilla",
                "-e",
                f"cat(if (requireNamespace('{package}', quietly=TRUE)) 'available' else 'not_installed')",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=12, check=False)
            status = (completed.stdout or completed.stderr or f"exit_{completed.returncode}").strip()
            available = status == "available"
        rows.append(
            {
                "language": "R",
                "package": package,
                "available": available,
                "use_in_this_isolation": "not_used_native_package",
            }
        )
    table = pd.DataFrame(rows)
    table["analysis_decision"] = np.where(
        table["available"],
        "available_but_not_required_for_resource_level_isolation",
        "unavailable_native_package_recorded",
    )
    return table


def bh_fdr(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    n = len(ranked)
    if n:
        adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
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


def paired_effect_statistics(table: pd.DataFrame, feature: str, shared_col: str, other_col: str) -> dict:
    paired = table.dropna(subset=[shared_col, other_col]).copy()
    diff = paired[shared_col].to_numpy(float) - paired[other_col].to_numpy(float)
    if len(diff) and not np.allclose(diff, 0):
        wilcoxon_p = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    else:
        wilcoxon_p = np.nan
    low, high = bootstrap_mean_ci(diff)
    return {
        "feature": feature,
        "patients": int(paired["sample"].nunique()),
        "positive_patients": int((diff > 0).sum()),
        "median_shared": float(paired[shared_col].median()) if len(paired) else np.nan,
        "median_other": float(paired[other_col].median()) if len(paired) else np.nan,
        "mean_difference_shared_minus_other": float(diff.mean()) if len(diff) else np.nan,
        "median_difference_shared_minus_other": float(np.median(diff)) if len(diff) else np.nan,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "wilcoxon_p": wilcoxon_p,
    }


def fetch_omnipath_resource() -> pd.DataFrame:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    cache = RESOURCES / "omnipath_ligrecextra_interactions.tsv"
    if not cache.exists():
        context = ssl.create_default_context()
        request = urllib.request.Request(OMNIPATH_URL, headers={"User-Agent": "pvr-pdr-ccs-isolation/1.0"})
        last_error: Exception | None = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=45, context=context) as response:
                    cache.write_bytes(response.read())
                break
            except Exception as exc:  # pragma: no cover - network fallback
                last_error = exc
        if not cache.exists() and last_error is not None:
            raise RuntimeError(f"Failed to download OmniPath resource: {last_error}") from last_error
    table = pd.read_csv(cache, sep="\t")
    return table


def build_lr_prior(omnipath: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    axis_definitions = []
    for axis, spec in BASE.LR_AXES.items():
        ligands = sorted(set(spec["ligands"]))
        receptors = sorted(set(spec["receptors"]))
        axis_definitions.append(
            {
                "axis": axis,
                "ligands": ";".join(ligands),
                "receptors": ";".join(receptors),
                "axis_definition": "prespecified compact axis from CCS signaling analysis",
            }
        )
        for ligand in ligands:
            for receptor in receptors:
                rows.append(
                    {
                        "axis": axis,
                        "ligand": ligand,
                        "receptor": receptor,
                        "resource_support": "manual_axis_pair",
                        "omnipath_sources": "",
                        "omnipath_references": "",
                        "curation_effort": np.nan,
                    }
                )

    prior = pd.DataFrame(rows)
    omni_rows = []
    for axis, spec in BASE.LR_AXES.items():
        ligands = set(spec["ligands"])
        receptors = set(spec["receptors"])
        sub = omnipath[
            omnipath["source_genesymbol"].isin(ligands)
            & omnipath["target_genesymbol"].isin(receptors)
        ].copy()
        for row in sub.itertuples(index=False):
            omni_rows.append(
                {
                    "axis": axis,
                    "ligand": row.source_genesymbol,
                    "receptor": row.target_genesymbol,
                    "resource_support": "omnipath_ligrecextra",
                    "omnipath_sources": getattr(row, "sources", ""),
                    "omnipath_references": getattr(row, "references", ""),
                    "curation_effort": getattr(row, "curation_effort", np.nan),
                }
            )
    if omni_rows:
        prior = pd.concat([prior, pd.DataFrame(omni_rows)], ignore_index=True)

    def combine(values: pd.Series) -> str:
        items = []
        for value in values.dropna().astype(str):
            if value and value != "nan":
                items.extend(part for part in value.split(";") if part)
        return ";".join(sorted(set(items)))

    prior = (
        prior.groupby(["axis", "ligand", "receptor"], as_index=False)
        .agg(
            resource_support=("resource_support", combine),
            omnipath_sources=("omnipath_sources", combine),
            omnipath_references=("omnipath_references", combine),
            curation_effort=("curation_effort", "max"),
        )
        .sort_values(["axis", "ligand", "receptor"])
    )
    prior["has_omnipath_support"] = prior["resource_support"].str.contains("omnipath", na=False)
    return prior, pd.DataFrame(axis_definitions)


def load_all_cell_metadata() -> pd.DataFrame:
    metas = []
    for accession, samples in FIB.COHORTS.items():
        annotations = pd.read_csv(ROOT / "results" / "formal_clustering" / accession / "final_singlet_annotations.csv.gz")
        for sample in samples:
            meta = annotations[annotations["sample"] == sample][["sample", "barcode", "final_annotation"]].copy()
            meta.insert(0, "accession", accession)
            meta["disease"] = FIB.disease_for(sample)
            metas.append(meta)
    return pd.concat(metas, ignore_index=True)


def load_target_state_metadata(all_meta: pd.DataFrame) -> pd.DataFrame:
    states = pd.read_csv(STATE_PATH)
    state_cols = states[["sample", "barcode", "state_annotation"]].copy()
    return all_meta.merge(state_cols, on=["sample", "barcode"], how="inner")


def summarize_group_expression_sparse(
    genes: list[str],
    meta: pd.DataFrame,
    group_cols: list[str],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = list(dict.fromkeys(genes))
    rows = []
    coverage = []
    for sample, sample_meta_original in meta.groupby("sample", sort=False):
        sample_meta = sample_meta_original.reset_index(drop=True)
        matrix_path = ROOT / "results" / "formal_qc" / "filtered_matrices" / sample / "counts_gene_by_cell.npz"
        genes_path = ROOT / "results" / "formal_qc" / "filtered_matrices" / sample / "genes.tsv.gz"
        barcodes_path = ROOT / "results" / "formal_qc" / "filtered_matrices" / sample / "barcodes.tsv.gz"
        gene_list = read_vector(genes_path)
        barcode_list = read_vector(barcodes_path)
        gene_index = {gene: index for index, gene in enumerate(gene_list)}
        barcode_index = {barcode: index for index, barcode in enumerate(barcode_list)}
        valid = sample_meta["barcode"].isin(barcode_index)
        sample_meta = sample_meta[valid].reset_index(drop=True)
        cell_indices = [barcode_index[barcode] for barcode in sample_meta["barcode"]]
        present = [gene for gene in genes if gene in gene_index]
        missing = [gene for gene in genes if gene not in gene_index]
        coverage.append(
            {
                "label": label,
                "sample": sample,
                "requested_genes": len(genes),
                "present_genes": len(present),
                "missing_gene_symbols": ";".join(missing),
            }
        )
        if sample_meta.empty:
            continue
        matrix = sparse.load_npz(matrix_path).tocsr()
        totals = np.asarray(matrix[:, cell_indices].sum(axis=0)).ravel()
        if present:
            raw = matrix[[gene_index[gene] for gene in present]][:, cell_indices].T.tocsr()
            raw = raw.multiply(np.divide(10000, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)[:, None])
            raw.data = np.log1p(raw.data)
            raw = raw.tocsr()
        else:
            raw = sparse.csr_matrix((len(cell_indices), 0), dtype=float)
        for group_key, positions in sample_meta.groupby(group_cols, sort=False).indices.items():
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            base = dict(zip(group_cols, group_key))
            sub = raw[list(positions), :]
            cells = int(len(positions))
            if present:
                means = np.asarray(sub.mean(axis=0)).ravel()
                fracs = np.asarray(sub.getnnz(axis=0), dtype=float) / max(cells, 1)
                detected = np.asarray(sub.getnnz(axis=0), dtype=int)
                for gene, mean, frac, det in zip(present, means, fracs, detected):
                    rows.append(
                        {
                            **base,
                            "gene": gene,
                            "cells": cells,
                            "expressing_cells": int(det),
                            "expressing_fraction": float(frac),
                            "mean_log_expression": float(mean),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def expression_lookup(table: pd.DataFrame, group_cols: list[str]) -> dict[tuple, dict]:
    lookup = {}
    for row in table.itertuples(index=False):
        key = tuple(getattr(row, col) for col in group_cols) + (row.gene,)
        lookup[key] = {
            "cells": int(row.cells),
            "expressing_fraction": float(row.expressing_fraction),
            "mean_log_expression": float(row.mean_log_expression),
            "expressing_cells": int(row.expressing_cells),
        }
    return lookup


def missing_expr() -> dict:
    return {"cells": 0, "expressing_fraction": 0.0, "mean_log_expression": 0.0, "expressing_cells": 0}


def score_lr_pairs(
    prior: pd.DataFrame,
    sender_summary: pd.DataFrame,
    receiver_summary: pd.DataFrame,
    all_meta: pd.DataFrame,
    target_meta: pd.DataFrame,
) -> pd.DataFrame:
    sender_cols = ["accession", "disease", "sample", "final_annotation"]
    receiver_cols = ["accession", "disease", "sample", "state_annotation"]
    sender_lookup = expression_lookup(sender_summary, sender_cols)
    receiver_lookup = expression_lookup(receiver_summary, receiver_cols)

    sender_totals = sender_summary.groupby(["sample", "gene"])["mean_log_expression"].sum().to_dict()
    receiver_totals = receiver_summary.groupby(["sample", "gene"])["mean_log_expression"].sum().to_dict()
    sample_info = all_meta[["accession", "disease", "sample"]].drop_duplicates().sort_values("sample")
    sender_groups = all_meta[sender_cols].drop_duplicates()
    receiver_groups = target_meta[receiver_cols].drop_duplicates()
    rows = []
    for info in sample_info.itertuples(index=False):
        senders = sender_groups[sender_groups["sample"] == info.sample]["final_annotation"].drop_duplicates().tolist()
        states = receiver_groups[receiver_groups["sample"] == info.sample]["state_annotation"].drop_duplicates().tolist()
        for axis_row in prior.itertuples(index=False):
            for sender in senders:
                ligand_key = (info.accession, info.disease, info.sample, sender, axis_row.ligand)
                ligand_expr = sender_lookup.get(ligand_key, missing_expr())
                for state in states:
                    receptor_key = (info.accession, info.disease, info.sample, state, axis_row.receptor)
                    receptor_expr = receiver_lookup.get(receptor_key, missing_expr())
                    ligand_mean = ligand_expr["mean_log_expression"]
                    receptor_mean = receptor_expr["mean_log_expression"]
                    ligand_frac = ligand_expr["expressing_fraction"]
                    receptor_frac = receptor_expr["expressing_fraction"]
                    expression_product = ligand_mean * receptor_mean
                    detection_product = ligand_frac * receptor_frac
                    passed_gate = (
                        ligand_expr["cells"] >= 10
                        and receptor_expr["cells"] >= 10
                        and ligand_frac >= 0.05
                        and receptor_frac >= 0.05
                    )
                    ligand_specificity = ligand_mean / (sender_totals.get((info.sample, axis_row.ligand), 0.0) + 1e-9)
                    receptor_specificity = receptor_mean / (receiver_totals.get((info.sample, axis_row.receptor), 0.0) + 1e-9)
                    specificity_product = ligand_specificity * receptor_specificity
                    product_detection_score = expression_product * math.sqrt(detection_product) if passed_gate else 0.0
                    rows.append(
                        {
                            "accession": info.accession,
                            "disease": info.disease,
                            "sample": info.sample,
                            "sender_cell_type": sender,
                            "receiver_state": state,
                            "axis": axis_row.axis,
                            "ligand": axis_row.ligand,
                            "receptor": axis_row.receptor,
                            "resource_support": axis_row.resource_support,
                            "has_omnipath_support": bool(axis_row.has_omnipath_support),
                            "sender_cells": ligand_expr["cells"],
                            "receiver_cells": receptor_expr["cells"],
                            "ligand_mean_log_expression": ligand_mean,
                            "receptor_mean_log_expression": receptor_mean,
                            "ligand_expressing_fraction": ligand_frac,
                            "receptor_expressing_fraction": receptor_frac,
                            "expression_product": expression_product,
                            "detection_product": detection_product,
                            "specificity_product": specificity_product,
                            "product_detection_score": product_detection_score,
                            "passed_detection_gate": bool(passed_gate),
                        }
                    )
    scores = pd.DataFrame(rows)
    ranked = []
    for sample, sub in scores.groupby("sample", sort=False):
        sub = sub.copy()
        n = max(len(sub), 1)
        rank_cols = []
        for metric in ["product_detection_score", "detection_product", "specificity_product"]:
            rank_col = f"{metric}_rank_pct"
            sub[rank_col] = sub[metric].rank(method="average", ascending=False, pct=True)
            rank_cols.append(rank_col)
        sub["consensus_rank_pct"] = sub[rank_cols].mean(axis=1)
        sub["consensus_score"] = np.clip(1.0 - sub["consensus_rank_pct"] + (1.0 / n), 0.0, 1.0)
        ranked.append(sub)
    return pd.concat(ranked, ignore_index=True)


def axis_patient_best_scores(pair_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_best = (
        pair_scores.groupby(["accession", "disease", "sample", "axis", "receiver_state"], as_index=False)
        .agg(
            best_consensus_score=("consensus_score", "max"),
            best_product_detection_score=("product_detection_score", "max"),
            gated_pairs=("passed_detection_gate", "sum"),
        )
    )
    rows = []
    for (accession, disease, sample, axis), sub in state_best.groupby(["accession", "disease", "sample", "axis"]):
        shared = sub[sub["receiver_state"] == SHARED_STATE]
        other = sub[sub["receiver_state"] != SHARED_STATE]
        if shared.empty or other.empty:
            continue
        rows.append(
            {
                "accession": accession,
                "disease": disease,
                "sample": sample,
                "axis": axis,
                "shared_consensus_score": float(shared["best_consensus_score"].max()),
                "other_consensus_score": float(other["best_consensus_score"].max()),
                "difference_consensus_score": float(shared["best_consensus_score"].max() - other["best_consensus_score"].max()),
                "shared_product_detection_score": float(shared["best_product_detection_score"].max()),
                "other_product_detection_score": float(other["best_product_detection_score"].max()),
                "difference_product_detection_score": float(shared["best_product_detection_score"].max() - other["best_product_detection_score"].max()),
                "shared_gated_pairs": int(shared["gated_pairs"].sum()),
                "other_gated_pairs": int(other["gated_pairs"].sum()),
            }
        )
    best = pd.DataFrame(rows)
    stat_rows = []
    for axis, sub in best.groupby("axis"):
        stat = paired_effect_statistics(sub, axis, "shared_consensus_score", "other_consensus_score")
        stat["metric"] = "consensus_score"
        stat_rows.append(stat)
        stat = paired_effect_statistics(sub, axis, "shared_product_detection_score", "other_product_detection_score")
        stat["metric"] = "product_detection_score"
        stat_rows.append(stat)
    stats_table = pd.DataFrame(stat_rows)
    stats_table["wilcoxon_fdr"] = bh_fdr(stats_table["wilcoxon_p"])
    stats_table["main_text_claim_status"] = stats_table.apply(classify_lr_stat, axis=1)
    return best, stats_table


def classify_lr_stat(row: pd.Series) -> str:
    if (
        row["metric"] == "consensus_score"
        and row["positive_patients"] >= 12
        and row["mean_difference_shared_minus_other"] > 0
        and pd.notna(row["wilcoxon_fdr"])
        and row["wilcoxon_fdr"] < 0.05
    ):
        return "candidate_shared_state_specific_communication_claim"
    if row["mean_difference_shared_minus_other"] <= 0:
        return "boundary_or_negative_for_shared_state_specific_communication"
    return "exploratory_not_strong_enough"


def build_gene_background(all_meta: pd.DataFrame) -> pd.DataFrame:
    sums: dict[str, float] = defaultdict(float)
    detections: dict[str, float] = defaultdict(float)
    total_cells = 0
    for sample, sample_meta in all_meta.groupby("sample", sort=False):
        matrix = sparse.load_npz(ROOT / "results" / "formal_qc" / "filtered_matrices" / sample / "counts_gene_by_cell.npz").tocsr()
        genes = read_vector(ROOT / "results" / "formal_qc" / "filtered_matrices" / sample / "genes.tsv.gz")
        barcodes = read_vector(ROOT / "results" / "formal_qc" / "filtered_matrices" / sample / "barcodes.tsv.gz")
        barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
        cell_indices = [barcode_index[barcode] for barcode in sample_meta["barcode"] if barcode in barcode_index]
        if not cell_indices:
            continue
        sub = matrix[:, cell_indices].tocsr()
        gene_sums = np.asarray(sub.sum(axis=1)).ravel()
        gene_detect = np.asarray(sub.getnnz(axis=1), dtype=float)
        for gene, value, detected in zip(genes, gene_sums, gene_detect):
            sums[gene] += float(value)
            detections[gene] += float(detected)
        total_cells += len(cell_indices)
    table = pd.DataFrame(
        {
            "gene": list(sums),
            "mean_raw_count": [sums[gene] / max(total_cells, 1) for gene in sums],
            "detecting_fraction": [detections[gene] / max(total_cells, 1) for gene in sums],
        }
    )
    table = table[
        (table["mean_raw_count"] > 0)
        & (table["detecting_fraction"] >= 0.01)
        & ~table["gene"].str.upper().str.startswith(("MT-", "RPL", "RPS"))
    ].copy()
    table["expression_bin"] = pd.qcut(np.log1p(table["mean_raw_count"]), q=10, duplicates="drop", labels=False)
    return table.sort_values("gene")


def expression_matched_random_ligands(
    prior: pd.DataFrame,
    background: pd.DataFrame,
    n_iter: int = 80,
) -> tuple[pd.DataFrame, list[str]]:
    bg_by_gene = background.set_index("gene")
    eligible_by_bin = {
        bin_id: sub["gene"].tolist()
        for bin_id, sub in background.groupby("expression_bin", dropna=True)
    }
    all_lr_genes = set(prior["ligand"]) | set(prior["receptor"])
    rows = []
    random_gene_pool = set()
    for axis, sub in prior.groupby("axis"):
        ligands = sorted(set(sub["ligand"]))
        receptors = sorted(set(sub["receptor"]))
        for iteration in range(n_iter):
            sampled = []
            for ligand in ligands:
                if ligand in bg_by_gene.index:
                    bin_id = bg_by_gene.loc[ligand, "expression_bin"]
                    candidates = [
                        gene
                        for gene in eligible_by_bin.get(bin_id, [])
                        if gene not in all_lr_genes and gene not in sampled
                    ]
                else:
                    candidates = [gene for gene in background["gene"].tolist() if gene not in all_lr_genes and gene not in sampled]
                if not candidates:
                    continue
                sampled_gene = str(RNG.choice(candidates))
                sampled.append(sampled_gene)
                random_gene_pool.add(sampled_gene)
            rows.append(
                {
                    "axis": axis,
                    "iteration": iteration,
                    "n_real_ligands": len(ligands),
                    "n_random_ligands": len(sampled),
                    "random_ligands": ";".join(sampled),
                    "real_ligands": ";".join(ligands),
                    "real_receptors": ";".join(receptors),
                }
            )
    return pd.DataFrame(rows), sorted(random_gene_pool)


def random_control_analysis(
    prior: pd.DataFrame,
    real_best: pd.DataFrame,
    random_ligands: pd.DataFrame,
    random_sender_summary: pd.DataFrame,
    receiver_summary: pd.DataFrame,
    all_meta: pd.DataFrame,
) -> pd.DataFrame:
    sender_cols = ["accession", "disease", "sample", "final_annotation"]
    receiver_cols = ["accession", "disease", "sample", "state_annotation"]
    sender_lookup = expression_lookup(random_sender_summary, sender_cols)
    receiver_lookup = expression_lookup(receiver_summary, receiver_cols)
    sample_info = all_meta[["accession", "disease", "sample"]].drop_duplicates().sort_values("sample")
    sender_groups = all_meta[sender_cols].drop_duplicates()
    receiver_states = receiver_summary[receiver_cols].drop_duplicates()

    random_delta_rows = []
    for axis_row in random_ligands.itertuples(index=False):
        receptors = [gene for gene in axis_row.real_receptors.split(";") if gene]
        ligands = [gene for gene in axis_row.random_ligands.split(";") if gene]
        for info in sample_info.itertuples(index=False):
            senders = sender_groups[sender_groups["sample"] == info.sample]["final_annotation"].drop_duplicates().tolist()
            states = receiver_states[receiver_states["sample"] == info.sample]["state_annotation"].drop_duplicates().tolist()
            state_scores = defaultdict(float)
            for sender in senders:
                for ligand in ligands:
                    ligand_expr = sender_lookup.get((info.accession, info.disease, info.sample, sender, ligand), missing_expr())
                    for state in states:
                        for receptor in receptors:
                            receptor_expr = receiver_lookup.get((info.accession, info.disease, info.sample, state, receptor), missing_expr())
                            passed = (
                                ligand_expr["cells"] >= 10
                                and receptor_expr["cells"] >= 10
                                and ligand_expr["expressing_fraction"] >= 0.05
                                and receptor_expr["expressing_fraction"] >= 0.05
                            )
                            if not passed:
                                continue
                            score = (
                                ligand_expr["mean_log_expression"]
                                * receptor_expr["mean_log_expression"]
                                * math.sqrt(ligand_expr["expressing_fraction"] * receptor_expr["expressing_fraction"])
                            )
                            state_scores[state] = max(state_scores[state], score)
            if SHARED_STATE not in states:
                continue
            shared = state_scores.get(SHARED_STATE, 0.0)
            other = max([state_scores.get(state, 0.0) for state in states if state != SHARED_STATE] or [0.0])
            random_delta_rows.append(
                {
                    "axis": axis_row.axis,
                    "iteration": axis_row.iteration,
                    "sample": info.sample,
                    "random_difference_product_detection_score": shared - other,
                }
            )
    random_delta = pd.DataFrame(random_delta_rows)
    if random_delta.empty:
        return pd.DataFrame()

    actual = real_best.groupby("axis", as_index=False)["difference_product_detection_score"].mean()
    random_iter = (
        random_delta.groupby(["axis", "iteration"], as_index=False)["random_difference_product_detection_score"].mean()
        .rename(columns={"random_difference_product_detection_score": "random_mean_difference_product_detection_score"})
    )
    rows = []
    for axis, sub in random_iter.groupby("axis"):
        actual_value = actual.loc[actual["axis"] == axis, "difference_product_detection_score"]
        if actual_value.empty:
            continue
        actual_value = float(actual_value.iloc[0])
        random_values = sub["random_mean_difference_product_detection_score"].to_numpy(float)
        rows.append(
            {
                "axis": axis,
                "actual_mean_difference_product_detection_score": actual_value,
                "random_iterations": len(random_values),
                "random_mean": float(random_values.mean()),
                "random_ci_low": float(np.quantile(random_values, 0.025)),
                "random_ci_high": float(np.quantile(random_values, 0.975)),
                "empirical_upper_tail_p": float((1 + np.sum(random_values >= actual_value)) / (len(random_values) + 1)),
                "actual_percentile_vs_random": float((np.sum(random_values <= actual_value) + 0.5) / len(random_values)),
                "control_interpretation": "actual_not_above_expression_matched_random" if actual_value <= np.quantile(random_values, 0.95) else "actual_above_random_exploratory",
            }
        )
    return pd.DataFrame(rows)


def nichenet_style_explanation(pair_scores: pd.DataFrame, axis_stats: pd.DataFrame) -> pd.DataFrame:
    core_genes = set(pd.read_csv(CORE_TARGETS_PATH)["gene"].astype(str))
    marker_genes = set(pd.read_csv(MARKERS_PATH).head(50)["gene"].astype(str))
    target_programs = {
        "core_25_shared_state": core_genes,
        "top50_shared_state_markers": marker_genes,
        **BASE.PATHWAY_GENE_SETS,
        **TFMOD.TF_TARGET_GENE_SETS,
    }
    axis_target_map = {
        "TGFB_TGFBR": {"TGF_beta_SMAD", "SMAD2_3_TGFbeta_targets"},
        "POSTN_integrin": {"ECM_integrin_FAK", "ECM_remodeling", "YAP_TAZ_TEAD", "TEAD_YAP_TAZ_targets"},
        "FN1_integrin": {"ECM_integrin_FAK", "ECM_remodeling", "YAP_TAZ_TEAD", "TEAD_YAP_TAZ_targets"},
        "THBS_TGFB_activation": {"TGF_beta_SMAD", "SMAD2_3_TGFbeta_targets", "ECM_remodeling"},
        "PDGF_PDGFR": {"PDGF_axis", "AP1_JUN_FOS"},
        "VEGF_VEGFR": {"VEGF_axis", "Hypoxia_HIF"},
        "SPP1_CD44_integrin": {"ECM_integrin_FAK", "NFkB_inflammatory", "RELA_NFKB_targets"},
        "CXCL_CXCR": {"NFkB_inflammatory", "RELA_NFKB_targets"},
        "CCL_CCR": {"NFkB_inflammatory", "RELA_NFKB_targets"},
    }
    supported_features = set()
    pathway_stats_path = find_script("run_ccs_signaling_axis_analysis.py").parents[1] / "results" / "pathway_activity_shared_vs_other_statistics.csv"
    tf_stats_path = find_script("run_ccs_tf_receptor_availability_analysis.py").parents[1] / "results" / "tf_target_program_activity_shared_vs_other_statistics.csv"
    for path in [pathway_stats_path, tf_stats_path]:
        if path.exists():
            table = pd.read_csv(path)
            for row in table.itertuples(index=False):
                feature = getattr(row, "feature")
                positive = getattr(row, "positive_patients")
                mean_diff = getattr(row, "mean_difference_shared_minus_other")
                fdr = getattr(row, "wilcoxon_fdr")
                if positive >= 12 and mean_diff > 0 and pd.notna(fdr) and fdr < 0.05:
                    supported_features.add(feature)

    shared_pair_scores = pair_scores[pair_scores["receiver_state"] == SHARED_STATE]
    rows = []
    for (axis, ligand), sub in shared_pair_scores.groupby(["axis", "ligand"]):
        target_features = sorted(axis_target_map.get(axis, set()))
        target_genes = set().union(*(target_programs.get(feature, set()) for feature in target_features))
        supported = sorted(set(target_features) & supported_features)
        top_sender = (
            sub.sort_values("consensus_score", ascending=False)
            .groupby("sample")
            .head(1)["sender_cell_type"]
            .mode()
        )
        axis_stat = axis_stats[(axis_stats["feature"] == axis) & (axis_stats["metric"] == "consensus_score")]
        rows.append(
            {
                "axis": axis,
                "ligand": ligand,
                "shared_state_samples_with_gated_pair": int(sub[sub["passed_detection_gate"]]["sample"].nunique()),
                "median_shared_consensus_score": float(sub.groupby("sample")["consensus_score"].max().median()),
                "most_common_top_sender": str(top_sender.iloc[0]) if len(top_sender) else "",
                "prespecified_target_programs": ";".join(target_features),
                "supported_target_programs": ";".join(supported),
                "n_supported_target_programs": len(supported),
                "core_25_overlap_genes": ";".join(sorted(target_genes & core_genes)),
                "top50_marker_overlap_genes": ";".join(sorted(target_genes & marker_genes)),
                "axis_positive_patients": int(axis_stat["positive_patients"].iloc[0]) if not axis_stat.empty else np.nan,
                "axis_mean_delta": float(axis_stat["mean_difference_shared_minus_other"].iloc[0]) if not axis_stat.empty else np.nan,
                "interpretation": ligand_target_interpretation(axis, supported, axis_stat),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_supported_target_programs", "median_shared_consensus_score"], ascending=[False, False])


def ligand_target_interpretation(axis: str, supported: list[str], axis_stat: pd.DataFrame) -> str:
    if axis_stat.empty:
        return "target-program context only; no axis-level communication statistic"
    row = axis_stat.iloc[0]
    if row["main_text_claim_status"] == "candidate_shared_state_specific_communication_claim":
        return "candidate communication claim if reviewed with spatial limitations"
    if supported:
        return "target-program context supported, but formal LR score is not shared-state specific"
    return "weak target-program and LR support; keep as boundary or do not emphasize"


def perturbation_support_table() -> pd.DataFrame:
    rows = []
    loo = ROOT / "results" / "reviewer_risk_validation" / "temporal_perturbation_leave_one_out.csv"
    if loo.exists():
        table = pd.read_csv(loo)
        for system, comparison in table[["system", "comparison"]].drop_duplicates().itertuples(index=False):
            sub = table[(table["system"] == system) & (table["comparison"] == comparison)]
            base = sub[sub["left_out"] == "none"]["effect"]
            sensitivity = sub[sub["left_out"] != "none"]["effect"]
            rows.append(
                {
                    "evidence_layer": "existing temporal or perturbation sensitivity",
                    "dataset_or_system": system,
                    "comparison": comparison,
                    "module_direction": float(base.iloc[0]) if not base.empty else np.nan,
                    "leave_one_min": float(sensitivity.min()) if not sensitivity.empty else np.nan,
                    "leave_one_max": float(sensitivity.max()) if not sensitivity.empty else np.nan,
                    "support_level": "directional_model_support",
                    "boundary": "not a therapeutic validation and not a physical communication assay",
                    "source_file": str(loo.relative_to(ROOT)),
                }
            )

    for path in ROOT.glob("*/results/gse282859_paired_emt_validation/paired_module_effects.csv"):
        table = pd.read_csv(path)
        for row in table.itertuples(index=False):
            rows.append(
                {
                    "evidence_layer": "external model-system perturbation/induction",
                    "dataset_or_system": "GSE282859 primary human RPE EMT",
                    "comparison": getattr(row, "module", "fixed core"),
                    "module_direction": getattr(row, "mean_paired_effect", getattr(row, "effect", np.nan)),
                    "leave_one_min": np.nan,
                    "leave_one_max": np.nan,
                    "support_level": "model_system_directional_support",
                    "boundary": "paired induction model; not PVR-specific and not treatment validation",
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    for path in ROOT.glob("*/results/gse244812_multimodal_validation/module_paired_effects.csv"):
        table = pd.read_csv(path)
        for row in table.itertuples(index=False):
            rows.append(
                {
                    "evidence_layer": "external model-system perturbation/induction",
                    "dataset_or_system": "GSE244812 multimodal RPE model",
                    "comparison": getattr(row, "module", "fixed core"),
                    "module_direction": getattr(row, "mean_paired_effect", getattr(row, "effect", np.nan)),
                    "leave_one_min": np.nan,
                    "leave_one_max": np.nan,
                    "support_level": "model_system_directional_support",
                    "boundary": "small paired model; use as directional support only",
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(rows)


def literature_spatial_immuno_support_table() -> pd.DataFrame:
    rows = [
        {
            "molecule_or_axis": "POSTN/periostin",
            "disease_context": "PDR fibrovascular membranes",
            "evidence_type": "vitreous and membrane expression; tissue-level support",
            "finding_summary": "Periostin was reported as increased in vitreous and fibrovascular membranes from PDR patients.",
            "source_title": "Increased expression of periostin in vitreous and fibrovascular membranes obtained from patients with proliferative diabetic retinopathy",
            "year": 2011,
            "pmid_or_doi": "PMID:21508107",
            "url": "https://pubmed.ncbi.nlm.nih.gov/21508107/",
            "manuscript_boundary": "literature-level tissue support only; not generated in this study",
        },
        {
            "molecule_or_axis": "POSTN/periostin",
            "disease_context": "PVR fibrous membrane model and human context",
            "evidence_type": "fibrous membrane formation evidence",
            "finding_summary": "Periostin was identified as a molecule involved in fibrous membrane formation in PVR-related work.",
            "source_title": "Periostin promotes the generation of fibrous membranes in proliferative vitreoretinopathy",
            "year": 2013,
            "pmid_or_doi": "PMID:24022401",
            "url": "https://pubmed.ncbi.nlm.nih.gov/24022401/",
            "manuscript_boundary": "supports prior biological plausibility, not a new mechanism from the present dataset",
        },
        {
            "molecule_or_axis": "COL1A1/COL3A1 collagen-rich ECM",
            "disease_context": "PVR preretinal membranes",
            "evidence_type": "immunohistochemical membrane characterization",
            "finding_summary": "PVR membrane stroma was reported to contain primarily collagen types I, II, and III.",
            "source_title": "Proliferative vitreoretinopathy membranes. An immunohistochemical study",
            "year": 1989,
            "pmid_or_doi": "PMID:2662102",
            "url": "https://pubmed.ncbi.nlm.nih.gov/2662102/",
            "manuscript_boundary": "supports ECM-rich membrane context, not shared-state-specific localization",
        },
        {
            "molecule_or_axis": "FN1/fibronectin and tenascin ECM",
            "disease_context": "PVR and PDR epiretinal membranes",
            "evidence_type": "immunohistochemical ECM comparison",
            "finding_summary": "Fibronectin and tenascin were reported as major ECM components in vitreoproliferative and diabetic epiretinal membranes.",
            "source_title": "Immunohistochemical study of extracellular matrix components in epiretinal membranes of vitreoproliferative retinopathy and proliferative diabetic retinopathy",
            "year": 2005,
            "pmid_or_doi": "PMID:15945009",
            "url": "https://pubmed.ncbi.nlm.nih.gov/15945009/",
            "manuscript_boundary": "literature-level ECM support; not evidence of a specific LR event",
        },
        {
            "molecule_or_axis": "FN1/fibronectin",
            "disease_context": "PDR epiretinal membranes",
            "evidence_type": "in situ hybridization and immunohistochemistry",
            "finding_summary": "Fibronectin mRNA and immunoreactivity were reported in PDR membranes.",
            "source_title": "The extracellular matrix of reparative tissue in the vitreous: fibronectin origin in diabetic preretinal membranes",
            "year": 1993,
            "pmid_or_doi": "Eye 1993; Nature link",
            "url": "https://www.nature.com/articles/eye199362.pdf",
            "manuscript_boundary": "supports membrane ECM localization from literature only",
        },
        {
            "molecule_or_axis": "THBS1/FN1 matricellular ECM",
            "disease_context": "epiretinal membranes",
            "evidence_type": "immunohistochemical colocalization",
            "finding_summary": "Thrombospondin and cellular fibronectin immunoreactivity were reported in most examined epiretinal membranes.",
            "source_title": "Thrombospondin as a component of the extracellular matrix of epiretinal membranes: comparisons with cellular fibronectin",
            "year": 1992,
            "pmid_or_doi": "PMID:1289132",
            "url": "https://pubmed.ncbi.nlm.nih.gov/1289132/",
            "manuscript_boundary": "supports ECM-rich tissue context; not shared-state-specific communication",
        },
        {
            "molecule_or_axis": "CTGF/TGF-beta-related fibrosis context",
            "disease_context": "PVR",
            "evidence_type": "intraocular fibrosis mediator study",
            "finding_summary": "CTGF was investigated as a mediator in PVR-related intraocular fibrosis.",
            "source_title": "Connective tissue growth factor as a mediator of intraocular fibrosis",
            "year": 2008,
            "pmid_or_doi": "PMID:18450591",
            "url": "https://pubmed.ncbi.nlm.nih.gov/18450591/",
            "manuscript_boundary": "TGF/CTGF plausibility only; not direct TGF receptor activation evidence",
        },
        {
            "molecule_or_axis": "COL1A1/TNC/CTGF",
            "disease_context": "proliferative retinal disease fibrovascular membranes",
            "evidence_type": "mRNA and immunohistochemical correlation",
            "finding_summary": "CTGF mRNA was studied in relation to type I collagen and tenascin immunohistochemical features.",
            "source_title": "Human connective tissue growth factor mRNA expression in fibrovascular membranes of proliferative retinal diseases",
            "year": 2002,
            "pmid_or_doi": "PMID:12207135",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12207135/",
            "manuscript_boundary": "supports fibrosis signaling context, not a new spatial experiment",
        },
    ]
    return pd.DataFrame(rows)


def write_summary(
    prior: pd.DataFrame,
    axis_best: pd.DataFrame,
    axis_stats: pd.DataFrame,
    random_control: pd.DataFrame,
    nichenet_style: pd.DataFrame,
    perturbation: pd.DataFrame,
    literature: pd.DataFrame,
) -> None:
    consensus = axis_stats[axis_stats["metric"] == "consensus_score"].copy()
    claim_ready = consensus[
        consensus["main_text_claim_status"] == "candidate_shared_state_specific_communication_claim"
    ]
    negative = consensus[
        consensus["main_text_claim_status"] == "boundary_or_negative_for_shared_state_specific_communication"
    ]
    lines = [
        "# Formal communication isolation summary",
        "",
        "## Bottom line",
        "",
        f"- LR prior pairs tested: {len(prior)} across {prior['axis'].nunique()} prespecified axes.",
        f"- OmniPath-supported prior pairs: {int(prior['has_omnipath_support'].sum())}/{len(prior)}.",
        f"- Axes meeting the conservative shared-state-specific communication rule: {len(claim_ready)}/{len(consensus)}.",
        f"- Axes with boundary/negative direction for shared-state-specific communication: {len(negative)}/{len(consensus)}.",
        "- This folder should be treated as isolated exploratory support. It does not replace the current manuscript claim unless reviewed and promoted.",
        "",
        "## Interpretation rule",
        "",
        "A formal communication claim requires patient-level shared-state enrichment, not merely ligand expression in a sender cell type and receptor expression in the shared state. If an axis fails this rule, the safest wording is that the shared state is better supported as a signaling-state convergence than as a proven physical cell-cell communication event.",
        "",
        "## Axis-level formal LR statistics",
        "",
        consensus[
            [
                "feature",
                "patients",
                "positive_patients",
                "mean_difference_shared_minus_other",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "wilcoxon_fdr",
                "main_text_claim_status",
            ]
        ].to_markdown(index=False),
        "",
        "## Random ligand control",
        "",
        random_control.to_markdown(index=False) if not random_control.empty else "Random control was not available.",
        "",
        "## NicheNet-style target-program interpretation",
        "",
        nichenet_style[
            [
                "axis",
                "ligand",
                "shared_state_samples_with_gated_pair",
                "median_shared_consensus_score",
                "most_common_top_sender",
                "supported_target_programs",
                "interpretation",
            ]
        ].head(30).to_markdown(index=False),
        "",
        "## Model perturbation support boundary",
        "",
        f"- Perturbation/model support rows collected: {len(perturbation)}.",
        "- Treat these rows as model-system directional support only, not therapeutic validation.",
        "",
        "## Literature-level spatial/immuno support boundary",
        "",
        f"- Literature support rows collected: {len(literature)}.",
        "- These rows are useful for a supplementary evidence table or Discussion sentence, but they are not generated by this study.",
    ]
    (RESULTS / "formal_communication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    RESOURCES.mkdir(parents=True, exist_ok=True)

    input_paths = [
        STATE_PATH,
        CORE_TARGETS_PATH,
        MARKERS_PATH,
        find_script("run_ccs_signaling_axis_analysis.py"),
        find_script("run_ccs_tf_receptor_availability_analysis.py"),
    ]
    write_manifest(input_paths, RESULTS / "input_manifest.csv")

    omnipath = fetch_omnipath_resource()
    package_status = package_feasibility_table()
    package_status.to_csv(RESULTS / "package_feasibility.csv", index=False)
    prior, prior_axis_def = build_lr_prior(omnipath)
    prior.to_csv(RESULTS / "formal_lr_prior_pairs.csv", index=False)
    prior_axis_def.to_csv(RESULTS / "formal_lr_axis_definitions.csv", index=False)

    all_meta = load_all_cell_metadata()
    target_meta = load_target_state_metadata(all_meta)
    lr_genes = sorted(set(prior["ligand"]) | set(prior["receptor"]))

    sender_summary, sender_coverage = summarize_group_expression_sparse(
        lr_genes,
        all_meta,
        ["accession", "disease", "sample", "final_annotation"],
        "lr_sender",
    )
    receiver_summary, receiver_coverage = summarize_group_expression_sparse(
        lr_genes,
        target_meta,
        ["accession", "disease", "sample", "state_annotation"],
        "lr_receiver",
    )
    pd.concat([sender_coverage, receiver_coverage], ignore_index=True).to_csv(
        RESULTS / "formal_lr_gene_coverage_by_sample.csv",
        index=False,
    )
    sender_summary.to_csv(RESULTS / "formal_sender_ligand_expression_by_patient_celltype.csv", index=False)
    receiver_summary.to_csv(RESULTS / "formal_receiver_receptor_expression_by_patient_state.csv", index=False)

    pair_scores = score_lr_pairs(prior, sender_summary, receiver_summary, all_meta, target_meta)
    pair_scores.to_csv(RESULTS / "formal_lr_pair_scores.csv.gz", index=False)
    axis_best, axis_stats = axis_patient_best_scores(pair_scores)
    axis_best.to_csv(RESULTS / "formal_lr_axis_patient_best_scores.csv", index=False)
    axis_stats.to_csv(RESULTS / "formal_lr_axis_shared_vs_other_statistics.csv", index=False)

    background = build_gene_background(all_meta)
    background.to_csv(RESULTS / "expression_background_for_random_ligands.csv", index=False)
    random_ligand_sets, random_gene_pool = expression_matched_random_ligands(prior, background)
    random_ligand_sets.to_csv(RESULTS / "expression_matched_random_ligand_sets.csv", index=False)
    if random_gene_pool:
        random_sender_summary, random_sender_coverage = summarize_group_expression_sparse(
            random_gene_pool,
            all_meta,
            ["accession", "disease", "sample", "final_annotation"],
            "random_sender_ligands",
        )
        random_sender_coverage.to_csv(RESULTS / "random_ligand_gene_coverage_by_sample.csv", index=False)
        random_control = random_control_analysis(
            prior,
            axis_best,
            random_ligand_sets,
            random_sender_summary,
            receiver_summary,
            all_meta,
        )
    else:
        random_control = pd.DataFrame()
    random_control.to_csv(RESULTS / "formal_lr_random_ligand_control.csv", index=False)

    nichenet_style = nichenet_style_explanation(pair_scores, axis_stats)
    nichenet_style.to_csv(RESULTS / "nichenet_style_ligand_target_explanation.csv", index=False)

    perturbation = perturbation_support_table()
    perturbation.to_csv(RESULTS / "model_perturbation_support_boundary.csv", index=False)
    literature = literature_spatial_immuno_support_table()
    literature.to_csv(RESULTS / "literature_spatial_immuno_support.csv", index=False)

    write_summary(prior, axis_best, axis_stats, random_control, nichenet_style, perturbation, literature)
    output_paths = sorted(path for path in RESULTS.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    output_paths += sorted(path for path in RESOURCES.iterdir() if path.is_file())
    write_manifest(output_paths, RESULTS / "output_manifest.csv")
    print(f"Formal communication isolation complete. Outputs: {RESULTS}")


if __name__ == "__main__":
    main()
