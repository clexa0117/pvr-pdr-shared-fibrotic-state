import gzip
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "reviewer_risk_validation"
FINAL_EVIDENCE = ROOT / "final_submission_materials/03_"
STAGE = ROOT / "legacy_archive/2026-06-14_/fibrotic_stage_target_exploration"
RNG = np.random.default_rng(20260615)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIB = load_module(ROOT / "scripts/run_fibrotic_state_analysis.py", "fib")
EXT = load_module(ROOT / "scripts/run_external_validation_targets.py", "ext")


def bootstrap_effect(left, right, iterations=1000):
    left, right = np.asarray(left, float), np.asarray(right, float)
    observed = EXT.hedges_g(left, right)
    values = []
    for _ in range(iterations):
        values.append(EXT.hedges_g(RNG.choice(left, len(left), True), RNG.choice(right, len(right), True)))
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def target_data():
    counts, genes, meta = FIB.load_target_cells()
    states = pd.read_csv(RESULTS / "fibrotic_state_analysis/fibrotic_cell_states.csv.gz")
    labels = states.set_index(["sample", "barcode"])["state_annotation"]
    meta["state_annotation"] = [labels.loc[(r.sample, r.barcode)] for r in meta.itertuples()]
    return counts, genes, meta, FIB.normalize_log(counts)


def select_blind_features(lognorm, genes, n=600):
    core = set(pd.read_csv(RESULTS / "patient_pseudobulk/robust_shared_state_genes.csv")["gene"])
    excluded = set().union(*FIB.PROGRAMS.values()) | core
    mean = np.asarray(lognorm.mean(axis=0)).ravel()
    var = np.asarray(lognorm.power(2).mean(axis=0)).ravel() - mean ** 2
    eligible = np.array([
        i for i, gene in enumerate(genes)
        if gene not in excluded and not gene.startswith(("MT-", "RPS", "RPL")) and mean[i] > 0.02
    ])
    return eligible[np.argsort(var[eligible])[-min(n, len(eligible)):]]


def classifier_validation(lognorm, genes, meta):
    features = select_blind_features(lognorm, genes)
    x = lognorm[:, features].toarray()
    y = (meta["state_annotation"] == "Shared_ECM_TGFB_high").astype(int).to_numpy()
    rows, models = [], {}
    for train in sorted(meta["accession"].unique()):
        train_mask = meta["accession"].eq(train).to_numpy()
        model = LogisticRegression(max_iter=1500, class_weight="balanced", C=0.3, random_state=0)
        model.fit(x[train_mask], y[train_mask])
        models[train] = model
        for test in sorted(meta["accession"].unique()):
            if test == train:
                continue
            mask = meta["accession"].eq(test).to_numpy()
            prob = model.predict_proba(x[mask])[:, 1]
            rows.append({
                "train_cohort": train, "test_cohort": test, "test_cells": int(mask.sum()),
                "test_shared_cells": int(y[mask].sum()), "roc_auc": roc_auc_score(y[mask], prob),
                "average_precision": average_precision_score(y[mask], prob),
                "balanced_accuracy_0_5": balanced_accuracy_score(y[mask], prob >= 0.5),
                "feature_count": len(features),
                "excluded_prespecified_program_and_core_genes": True,
            })
    pd.DataFrame(rows).to_csv(OUT / "cross_cohort_blind_classifier.csv", index=False)
    pd.DataFrame({"gene": np.asarray(genes)[features]}).to_csv(OUT / "blind_classifier_features.csv", index=False)
    return models, np.asarray(genes)[features]


def load_all_cells(feature_genes):
    matrices, metas = [], []
    for accession, samples in FIB.COHORTS.items():
        annotations = pd.read_csv(RESULTS / f"formal_clustering/{accession}/final_singlet_annotations.csv.gz")
        for sample in samples:
            selected = annotations[annotations["sample"] == sample].copy()
            genes = FIB.read_vector(RESULTS / f"formal_qc/filtered_matrices/{sample}/genes.tsv.gz")
            barcodes = FIB.read_vector(RESULTS / f"formal_qc/filtered_matrices/{sample}/barcodes.tsv.gz")
            matrix = sparse.load_npz(RESULTS / f"formal_qc/filtered_matrices/{sample}/counts_gene_by_cell.npz").tocsr()
            gene_idx, barcode_idx = {g: i for i, g in enumerate(genes)}, {b: i for i, b in enumerate(barcodes)}
            rows = [gene_idx[g] for g in feature_genes]
            cols = [barcode_idx[b] for b in selected["barcode"]]
            matrices.append(matrix[rows][:, cols].T.tocsr())
            selected = selected[["sample", "barcode", "final_annotation"]]
            selected.insert(0, "accession", accession)
            selected["disease"] = FIB.disease_for(sample)
            metas.append(selected)
    counts = sparse.vstack(matrices).tocsr()
    return FIB.normalize_log(counts).toarray(), pd.concat(metas, ignore_index=True)


def whole_atlas_prediction(models, feature_genes):
    x, meta = load_all_cells(feature_genes)
    probs = np.column_stack([model.predict_proba(x)[:, 1] for model in models.values()])
    meta["blind_consensus_probability"] = np.median(probs, axis=1)
    meta["blind_consensus_positive"] = meta["blind_consensus_probability"] >= 0.5
    meta["preselected_fibrosis_related_type"] = meta["final_annotation"].isin(FIB.TARGET_TYPES)
    summary = meta.groupby(
        ["disease", "preselected_fibrosis_related_type", "final_annotation"], as_index=False
    ).agg(cells=("barcode", "size"), predicted_positive=("blind_consensus_positive", "sum"),
          median_probability=("blind_consensus_probability", "median"))
    summary["predicted_positive_fraction"] = summary["predicted_positive"] / summary["cells"]
    summary.to_csv(OUT / "whole_atlas_blind_prediction_by_cell_type.csv", index=False)
    patient = meta.groupby(["accession", "disease", "sample"], as_index=False).agg(
        cells=("barcode", "size"), predicted_positive=("blind_consensus_positive", "sum"),
        non_preselected_predicted_positive=("blind_consensus_positive", lambda x: 0),
    )
    non_target = meta[~meta["preselected_fibrosis_related_type"]].groupby("sample")["blind_consensus_positive"].sum()
    patient["non_preselected_predicted_positive"] = patient["sample"].map(non_target).fillna(0).astype(int)
    patient.to_csv(OUT / "whole_atlas_blind_prediction_by_patient.csv", index=False)


def patient_within_state_enrichment(lognorm, genes, meta):
    core = pd.read_csv(RESULTS / "patient_pseudobulk/robust_shared_state_genes.csv")["gene"].tolist()
    index = {g: i for i, g in enumerate(genes)}
    core_idx = [index[g] for g in core if g in index]
    module = np.asarray(lognorm[:, core_idx].mean(axis=1)).ravel()
    shared = meta["state_annotation"].eq("Shared_ECM_TGFB_high").to_numpy()
    rows = []
    gene_rows = []
    for sample in sorted(meta["sample"].unique()):
        sm = meta["sample"].eq(sample).to_numpy()
        left, right = module[sm & shared], module[sm & ~shared]
        effect, low, high = bootstrap_effect(left, right)
        rows.append({
            "sample": sample, "disease": meta.loc[sm, "disease"].iloc[0],
            "shared_cells": len(left), "other_state_cells": len(right),
            "mean_difference": float(left.mean() - right.mean()), "hedges_g": effect,
            "bootstrap_ci_low": low, "bootstrap_ci_high": high, "positive_direction": effect > 0,
        })
        for gene in core:
            values = np.asarray(lognorm[:, index[gene]].todense()).ravel()
            gene_rows.append({
                "sample": sample, "disease": meta.loc[sm, "disease"].iloc[0], "gene": gene,
                "mean_difference_shared_vs_other": float(values[sm & shared].mean() - values[sm & ~shared].mean()),
            })
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT / "patient_within_state_core_module_effects.csv", index=False)
    gene_effects = pd.DataFrame(gene_rows)
    gene_effects.to_csv(OUT / "patient_within_state_gene_effects.csv", index=False)
    consistency = gene_effects.groupby("gene", as_index=False).agg(
        patients_positive=("mean_difference_shared_vs_other", lambda x: int((x > 0).sum())),
        patients_total=("sample", "nunique"), median_effect=("mean_difference_shared_vs_other", "median"),
    )
    consistency.to_csv(OUT / "patient_within_state_gene_direction_consistency.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    effects = effects.sort_values(["disease", "hedges_g"])
    colors = effects["disease"].map({"PVR": "#d95f5f", "PDR": "#4c78a8"})
    ax.errorbar(effects["hedges_g"], np.arange(len(effects)), xerr=[
        effects["hedges_g"] - effects["bootstrap_ci_low"], effects["bootstrap_ci_high"] - effects["hedges_g"]
    ], fmt="none", ecolor=colors, alpha=.8)
    ax.scatter(effects["hedges_g"], np.arange(len(effects)), c=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(np.arange(len(effects)), effects["sample"], fontsize=7)
    ax.set_xlabel("Hedges' g: shared state vs other fibrosis-related states")
    ax.set_title("Patient-within-state enrichment of the 25-gene core")
    fig.tight_layout()
    fig.savefig(OUT / "patient_within_state_core_module_effects.png", dpi=220)
    plt.close(fig)


def matched_random_sets(lognorm, genes, core, n_sets=100):
    mean = np.asarray(lognorm.mean(axis=0)).ravel()
    table = pd.DataFrame({"gene": genes, "mean": mean})
    table = table[~table["gene"].str.startswith(("MT-", "RPS", "RPL"))].copy()
    table["bin"] = pd.qcut(table["mean"].rank(method="first"), 10, labels=False)
    bins = dict(zip(table["gene"], table["bin"]))
    pools = {b: table.loc[table["bin"] == b, "gene"].tolist() for b in range(10)}
    excluded = set(core)
    sets = []
    for _ in range(n_sets):
        chosen = []
        for gene in core:
            pool = [g for g in pools[bins[gene]] if g not in excluded and g not in chosen]
            chosen.append(RNG.choice(pool))
        sets.append(set(chosen))
    return sets


def external_benchmark(lognorm, genes):
    core = set(pd.read_csv(RESULTS / "patient_pseudobulk/robust_shared_state_genes.csv")["gene"])
    generic = FIB.PROGRAMS["ECM"] | FIB.PROGRAMS["Contractile"] | FIB.PROGRAMS["TGFB_Mechanosensing"]
    random_sets = matched_random_sets(lognorm, genes, sorted(core))
    requested = core | generic | set().union(*random_sets)
    probe = EXT.annotation()
    datasets = [EXT.load_gse179603(requested), EXT.load_series("GSE60436", requested, probe), EXT.load_series("GSE41019", requested, probe)]
    rows = []
    for dataset in datasets:
        def score_for(module):
            present = sorted(module & set(dataset[2]))
            matrix = np.vstack([dataset[2][g] for g in present])
            sd = matrix.std(axis=1, keepdims=True); sd[sd == 0] = 1
            return ((matrix - matrix.mean(axis=1, keepdims=True)) / sd).mean(axis=0), len(present)

        for module_name, module in [("25_gene_core", core), ("generic_fibrosis", generic)]:
            for row in EXT.module_score(dataset, module):
                left_name, right_name = row["comparison"].split("_vs_")
                score, _ = score_for(module)
                effect, low, high = bootstrap_effect(score[dataset[3][left_name]], score[dataset[3][right_name]])
                rows.append({**row, "module": module_name, "ci_low": low, "ci_high": high})
        comparison = EXT.module_score(dataset, core)
        for core_result in comparison:
            random_g = []
            for random_set in random_sets:
                score, _ = score_for(random_set)
                left_name, right_name = core_result["comparison"].split("_vs_")
                random_g.append(EXT.hedges_g(score[dataset[3][left_name]], score[dataset[3][right_name]]))
            rows.append({
                **core_result, "module": "25_gene_core_random_benchmark",
                "ci_low": np.nan, "ci_high": np.nan,
                "random_median_hedges_g": float(np.median(random_g)),
                "random_95pct_hedges_g": float(np.quantile(random_g, .95)),
                "core_random_percentile": float(np.mean(np.asarray(random_g) <= core_result["hedges_g"])),
            })
    pd.DataFrame(rows).to_csv(OUT / "external_module_benchmark_and_ci.csv", index=False)


def model_leave_one_out():
    rows = []
    rabbit = pd.read_csv(FINAL_EVIDENCE / "rabbit_muller_sample_stage_scores.csv")
    nuclei = rabbit[rabbit["run"].str.lower().eq("nuclei")]
    for left, right in [("Hr4", "Control"), ("Day14", "Control"), ("Day14", "Hr4")]:
        for drop in ["none"] + nuclei["orig.ident"].tolist():
            data = nuclei if drop == "none" else nuclei[nuclei["orig.ident"] != drop]
            a, b = data.loc[data["treatment"] == left, "shared_core_score"], data.loc[data["treatment"] == right, "shared_core_score"]
            if len(a) and len(b):
                rows.append({"system": "rabbit_nuclei", "comparison": f"{left}_vs_{right}", "left_out": drop, "effect": a.mean() - b.mean()})
    inhibitor = pd.read_csv(FINAL_EVIDENCE / "independent_human_inhibitor_module_scores.csv")
    for condition in ["bay16", "bay32"]:
        for drop in ["none"] + inhibitor["sample"].tolist():
            data = inhibitor if drop == "none" else inhibitor[inhibitor["sample"] != drop]
            a = data.loc[data["condition"] == condition, "mean_core_fpkm"]
            b = data.loc[data["condition"] == "induced", "mean_core_fpkm"]
            if len(a) and len(b):
                rows.append({"system": "independent_inhibitor", "comparison": f"{condition}_vs_induced", "left_out": drop, "effect": a.mean() - b.mean()})
    rpe = pd.read_csv(FINAL_EVIDENCE / "human_rpe_time_intervention_module_scores.csv")
    for time in ["8h", "24h"]:
        for left, right, label in [("TNT", "Media", "induction"), ("TNT_plus_Polymer", "TNT", "polymer_reduction")]:
            sub = rpe[rpe["time"] == time]
            pairs = sorted(set(sub.loc[sub["condition"] == left, "replicate"]) & set(sub.loc[sub["condition"] == right, "replicate"]))
            for drop in ["none"] + pairs:
                keep = pairs if drop == "none" else [p for p in pairs if p != drop]
                effects = []
                for pair in keep:
                    lv = sub[(sub["condition"] == left) & (sub["replicate"] == pair)]["shared_core_score"].iloc[0]
                    rv = sub[(sub["condition"] == right) & (sub["replicate"] == pair)]["shared_core_score"].iloc[0]
                    effects.append(lv - rv)
                rows.append({"system": "human_rpe_paired", "comparison": f"{label}_{time}", "left_out": drop, "effect": float(np.mean(effects))})
    pd.DataFrame(rows).to_csv(OUT / "temporal_perturbation_leave_one_out.csv", index=False)


def candidate_evidence_matrix():
    data = pd.read_csv(RESULTS / "mechanism_evidence_integration/target_integrated_evidence.csv")
    score = pd.DataFrame({"gene": data["gene"]})
    score["state_specificity"] = data["expressing_fraction_difference"] >= .30
    score["all_patient_detection"] = data["patients_any_expression"] == 15
    score["cross_disease_external_direction"] = data["cross_disease_external_direction_support"].fillna(False)
    score["cross_disease_balance"] = data["cross_disease_balance_ratio"] >= .30
    score["direct_ocular_evidence"] = data["translational_tier"].eq("Tier_1_direct_ocular")
    score["no_explicit_safety_block"] = ~data["integrated_decision"].isin(["Risk_review", "Do_not_inhibit"])
    criteria = score.columns[1:]
    score["unweighted_evidence_count"] = score[criteria].sum(axis=1)
    score.to_csv(OUT / "candidate_transparent_evidence_matrix.csv", index=False)
    sensitivity = []
    for excluded in ["none"] + list(criteria):
        use = list(criteria) if excluded == "none" else [c for c in criteria if c != excluded]
        for row in score.itertuples():
            sensitivity.append({"excluded_criterion": excluded, "gene": row.gene, "evidence_count": sum(getattr(row, c) for c in use)})
    pd.DataFrame(sensitivity).to_csv(OUT / "candidate_ranking_sensitivity.csv", index=False)


def reporting_tables():
    qc = pd.read_csv(RESULTS / "formal_qc/sample_qc_summary.csv")
    annotations = pd.read_csv(RESULTS / "formal_clustering/sample_annotation_counts.csv")
    states = pd.read_csv(RESULTS / "fibrotic_state_analysis/patient_state_counts.csv")
    cell_counts = annotations.groupby(["accession", "sample"], as_index=False)["cells"].sum().rename(columns={"cells": "final_annotated_cells"})
    shared = states[states["state_annotation"] == "Shared_ECM_TGFB_high"][["sample", "size"]].rename(columns={"size": "shared_state_cells"})
    qc.merge(cell_counts, on=["accession", "sample"], how="left").merge(shared, on="sample", how="left").fillna({"shared_state_cells": 0}).to_csv(
        OUT / "sample_qc_and_state_reporting_table.csv", index=False
    )
    rows = [{"program": name, "gene": gene} for name, genes in FIB.PROGRAMS.items() for gene in sorted(genes)]
    pd.DataFrame(rows).to_csv(OUT / "program_gene_sets.csv", index=False)
    manifest = pd.read_csv(RESULTS / "reproducibility_freeze/formal_result_manifest.csv")
    manifest.to_csv(OUT / "complete_formal_source_manifest.csv", index=False)


def plot_directional_model_support():
    rabbit = pd.read_csv(FINAL_EVIDENCE / "rabbit_muller_sample_stage_scores.csv")
    human = pd.read_csv(FINAL_EVIDENCE / "human_rpe_time_intervention_module_scores.csv")
    inhibitor = pd.read_csv(FINAL_EVIDENCE / "independent_human_inhibitor_module_scores.csv")
    loo = pd.read_csv(OUT / "temporal_perturbation_leave_one_out.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    order = ["Control", "Hr4", "Day14"]
    for run, marker, color in [("Cells", "o", "#3B82F6"), ("Nuclei", "s", "#D97706")]:
        sub = rabbit[rabbit["run"] == run]
        for x, treatment in enumerate(order):
            vals = sub.loc[sub["treatment"] == treatment, "shared_core_score"]
            ax.scatter(np.full(len(vals), x), vals, marker=marker, color=color, s=55, alpha=.75, label=run if x == 0 else None)
            if len(vals):
                ax.hlines(vals.mean(), x - .18, x + .18, color=color, lw=2)
    ax.set_xticks(range(3), ["Control", "4 h", "14 d"])
    ax.set_ylabel("25-gene core score")
    ax.set_title("A. Independent rabbit samples (no longitudinal connection)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for condition, color in [("Media", "#6B7280"), ("TNT", "#DC2626"), ("TNT_plus_Polymer", "#2563EB")]:
        sub = human[human["condition"] == condition]
        means = sub.groupby("time")["shared_core_score"].mean().reindex(["8h", "24h"])
        ax.plot([0, 1], means, marker="o", linewidth=2, label=condition.replace("_", " "), color=color)
        for x, time in enumerate(["8h", "24h"]):
            vals = sub.loc[sub["time"] == time, "shared_core_score"]
            ax.scatter(np.full(len(vals), x), vals, color=color, alpha=.45, s=35)
    ax.set_xticks([0, 1], ["8 h", "24 h"])
    ax.set_ylabel("25-gene core score")
    ax.set_title("B. Paired human RPE directions (n=3 pairs)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    condition_order = ["untreated", "induced", "vehicle", "bay16", "bay32"]
    means = inhibitor.groupby("condition")["mean_core_fpkm"].mean().reindex(condition_order)
    ax.bar(range(len(means)), means, color=["#6B7280", "#DC2626", "#F59E0B", "#60A5FA", "#2563EB"])
    for x, condition in enumerate(condition_order):
        vals = inhibitor.loc[inhibitor["condition"] == condition, "mean_core_fpkm"]
        ax.scatter(np.full(len(vals), x), vals, color="black", s=28, alpha=.75)
    ax.set_xticks(range(len(means)), ["Untreated", "Induced", "Vehicle", "BAY16", "BAY32"], rotation=25, ha="right")
    ax.set_ylabel("Mean core FPKM")
    ax.set_title("C. IKK-beta inhibitor: high-dose effect is outlier-sensitive")

    ax = axes[1, 1]
    selected = loo[
        ((loo["system"] == "rabbit_nuclei") & loo["comparison"].isin(["Hr4_vs_Control", "Day14_vs_Control"])) |
        ((loo["system"] == "independent_inhibitor") & loo["comparison"].eq("bay32_vs_induced")) |
        ((loo["system"] == "human_rpe_paired") & loo["comparison"].isin(["induction_24h", "polymer_reduction_24h"]))
    ]
    labels = selected["system"] + "\n" + selected["comparison"]
    groups = list(dict.fromkeys(labels))
    for x, label in enumerate(groups):
        vals = selected.loc[labels == label, "effect"]
        ax.scatter(np.full(len(vals), x), vals, s=25, alpha=.6)
        ax.hlines(vals.iloc[0], x - .2, x + .2, color="black", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Effect after leave-one-sample/pair removal")
    ax.set_title("D. Directional sensitivity")
    fig.suptitle("Directional temporal and perturbation support with explicit small-sample sensitivity", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "figure_6_directional_model_support.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary():
    classifier = pd.read_csv(OUT / "cross_cohort_blind_classifier.csv")
    patient = pd.read_csv(OUT / "patient_within_state_core_module_effects.csv")
    benchmark = pd.read_csv(OUT / "external_module_benchmark_and_ci.csv")
    loo = pd.read_csv(OUT / "temporal_perturbation_leave_one_out.csv")
    lines = [
        "# Reviewer-risk validation summary", "",
        "## Circular-definition risk", "",
        f"- Cross-cohort classifiers used {int(classifier['feature_count'].iloc[0])} expression features after excluding all prespecified program genes and the 25-gene core.",
        f"- Blind cross-cohort ROC AUC range: {classifier['roc_auc'].min():.3f}-{classifier['roc_auc'].max():.3f}.",
        "- This validates transportability of the originally defined label; it is not presented as an independent de novo state discovery.",
        "",
        "## Patient-within-state enrichment", "",
        f"- The 25-gene module was directionally enriched in shared versus other fibrosis-related states in {int(patient['positive_direction'].sum())}/{len(patient)} patients.",
        "",
        "## External benchmark", "",
        "- External tissue effects are reported with bootstrap confidence intervals and compared with a generic fibrosis module and expression-matched random gene sets.",
        "- These analyses support directional program recurrence, not identity of the same cell state in bulk tissue.",
        "",
        "## Temporal and perturbation evidence", "",
        "- Leave-one-sample and leave-one-pair results are reported explicitly. Wording should remain directional model support rather than efficacy or statistically established early induction.",
        "",
        "## Candidate prioritization", "",
        "- Candidate evidence is reported as transparent unweighted criteria with leave-one-criterion-out sensitivity. No unique optimized weighting scheme is claimed.",
    ]
    (OUT / "reviewer_risk_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts, genes, meta, lognorm = target_data()
    models, feature_genes = classifier_validation(lognorm, genes, meta)
    whole_atlas_prediction(models, feature_genes)
    patient_within_state_enrichment(lognorm, genes, meta)
    external_benchmark(lognorm, genes)
    model_leave_one_out()
    candidate_evidence_matrix()
    reporting_tables()
    plot_directional_model_support()
    write_summary()


if __name__ == "__main__":
    main()
