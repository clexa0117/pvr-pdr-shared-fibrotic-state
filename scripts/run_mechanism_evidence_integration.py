from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "mechanism_evidence_integration"


def evidence_level(single_cell_support, patient_robustness, external_support):
    if single_cell_support and patient_robustness and external_support:
        return "Accepted"
    if single_cell_support and (patient_robustness or external_support):
        return "Supported_with_boundary"
    return "Exploratory_or_not_accepted"


def target_decision(translational_tier, shared_specificity, external_support):
    if translational_tier.startswith("Do_not_inhibit"):
        return "Do_not_inhibit"
    if translational_tier.startswith("Risk_review"):
        return "Risk_review"
    if translational_tier == "Tier_1_direct_ocular" and external_support:
        return "Primary_candidate"
    if translational_tier == "Tier_2_mechanistic" and shared_specificity > 0 and external_support:
        return "Mechanistic_candidate"
    if translational_tier == "Tier_3_emerging" and external_support:
        return "Emerging_candidate"
    return "Not_prioritized"


def build_claim_matrix():
    claims = [
        {
            "claim_id": "C1", "claim": "PVR and PDR retain different source-associated state directions.",
            "single_cell_support": True, "patient_robustness": True, "external_support": False,
            "evidence": "PVR enriched for Muller_stress/RPE_inflammatory; PDR enriched for Pericyte_contractile; patient-level origin programs differ.",
            "boundary": "Expression states do not prove lineage origin or transdifferentiation.",
            "manuscript_wording": "PVR and PDR retained distinct source-associated transcriptional features.",
        },
        {
            "claim_id": "C2", "claim": "Different source-associated cells converge on a shared high-fibrosis state.",
            "single_cell_support": True, "patient_robustness": True, "external_support": True,
            "evidence": "Shared_ECM_TGFB_high contains 2,335 cells, covers 6/6 PVR and 9/9 PDR patients, and alternative high-program states cover all patients.",
            "boundary": "Convergence is transcriptional-state convergence, not direct lineage tracing.",
            "manuscript_wording": "Distinct source-associated populations converged transcriptionally on a shared ECM/TGF-beta-high fibrotic state.",
        },
        {
            "claim_id": "C3", "claim": "A 25-gene shared core is detected across patients and enriched within the shared state.",
            "single_cell_support": True, "patient_robustness": True, "external_support": True,
            "evidence": "25/25 genes reach CPM >= 1 in 15/15 patients; the module is enriched in shared versus other fibrosis-related states in 15/15 patients; 25/25 retained after every patient/cohort exclusion.",
            "boundary": "Detection does not itself establish state specificity; several individual genes are not positively enriched in every patient.",
            "manuscript_wording": "A 25-gene core was detected across all patients and showed consistent patient-within-state module enrichment.",
        },
        {
            "claim_id": "C4", "claim": "The shared core is directionally validated in external PVR and PDR membrane tissues.",
            "single_cell_support": True, "patient_robustness": True, "external_support": True,
            "evidence": "The 25-gene module is positive in all four external comparisons; three are permutation-significant.",
            "boundary": "Bulk tissues mainly validate a general fibrosis direction; the core did not outperform expression-matched random modules in every comparison, and confidence intervals were wide in small datasets.",
            "manuscript_wording": "The shared core showed directional recurrence in independent membrane datasets, without establishing the identity of the same cell state.",
        },
        {
            "claim_id": "C4b", "claim": "An independent human PVR single-cell dataset localizes the fixed shared core to fibroblasts.",
            "single_cell_support": True, "patient_robustness": True, "external_support": True,
            "evidence": "In SCP2582, the fixed 25-gene core was higher in author-annotated fibroblasts than other PVR membrane cells in 8/8 donors, with mean donor-level effect 1.561 and an expression-matched random-set empirical upper-tail p = 0.0010.",
            "boundary": "This is independent cellular-context validation in processed PVR single-cell data; it does not test RUNX1 intervention response or prove that PVR and PDR contain an identical state.",
            "manuscript_wording": "An independent eight-donor human PVR single-cell object localized the fixed core to fibroblasts and exceeded expression-matched random gene sets.",
        },
        {
            "claim_id": "C5", "claim": "POSTN is an established candidate supported across the largest number of evaluated evidence layers.",
            "single_cell_support": True, "patient_robustness": True, "external_support": True,
            "evidence": "POSTN is shared-state enriched, detected in 15/15 patients, stable in sensitivity analyses, externally supported, and has direct ocular intervention evidence.",
            "boundary": "POSTN expression is substantially stronger in PDR than PVR; expression does not establish treatment efficacy.",
            "manuscript_wording": "POSTN was the most consistently supported established candidate across the evaluated evidence layers, although its expression was quantitatively PDR-biased.",
        },
        {
            "claim_id": "C6", "claim": "PVR/PDR-specific single-gene differences are established.",
            "single_cell_support": True, "patient_robustness": False, "external_support": False,
            "evidence": "Exploratory differences are numerous and confounded by cohort, source, and tissue composition.",
            "boundary": "No disease-specific single gene is formally accepted.",
            "manuscript_wording": "Disease-specific single-gene effects were treated as exploratory because disease and cohort were inseparable.",
        },
        {
            "claim_id": "C7", "claim": "A subset of shared extracellular candidates is repeatedly detectable in human ocular-fluid proteomics.",
            "single_cell_support": True, "patient_robustness": False, "external_support": True,
            "evidence": "FN1, IGFBP7, and TIMP1 are detected in four independent ocular-fluid proteomics projects; PXD077831 measures baseline vitreous before subsequent PVR.",
            "boundary": "Protein detection is not disease-specific elevation; PXD077831 has no FDR-significant proteins and does not validate predictive biomarkers.",
            "manuscript_wording": "Selected shared extracellular candidates showed repeated human ocular-fluid protein detection, including prospective baseline-vitreous support before subsequent PVR.",
        },
    ]
    result = pd.DataFrame(claims)
    result["evidence_level"] = result.apply(
        lambda row: evidence_level(row["single_cell_support"], row["patient_robustness"], row["external_support"]), axis=1
    )
    result.loc[result["claim_id"] == "C6", "evidence_level"] = "Exploratory_or_not_accepted"
    return result


def build_core_gene_evidence():
    core = pd.read_csv(RESULTS / "patient_pseudobulk/robust_shared_state_genes.csv")
    external = pd.read_csv(RESULTS / "external_validation_targets/external_evidence_matrix.csv")
    strict = pd.read_csv(RESULTS / "robustness_sensitivity/threshold_grid_shared_core.csv")
    strict_genes = set(strict[(strict["cpm_threshold"] == 2) & (strict["required_patient_fraction"] == 1)]["qualifying_genes"].iloc[0].split(";"))
    result = core.merge(external[[
        "gene", "target_class", "cross_disease_external_direction_support", "PVR_support_count",
        "GSE60436_PDR_FVM_vs_retina_support", "translational_tier",
    ]], on="gene", how="left")
    result["retained_CPM2_all_patients"] = result["gene"].isin(strict_genes)
    result["evidence_role"] = np.where(
        result["translational_tier"].fillna("Not_prioritized").ne("Not_prioritized"),
        "Core_and_target_evidence", "Shared_core_evidence",
    )
    return result


def build_target_evidence():
    external = pd.read_csv(RESULTS / "external_validation_targets/external_evidence_matrix.csv")
    cellular = pd.read_csv(RESULTS / "target_cellular_specificity/target_experimental_priority.csv")
    result = cellular.merge(external[[
        "gene", "cross_disease_external_direction_support", "PVR_support_count",
        "GSE60436_PDR_FVM_vs_retina_support", "mechanism_rationale", "literature_urls",
    ]], on="gene", how="left")
    result["integrated_decision"] = result.apply(lambda row: target_decision(
        row["translational_tier"],
        row["expressing_fraction_difference"],
        bool(row["cross_disease_external_direction_support"]),
    ), axis=1)
    result["key_boundary"] = result["gene"].map({
        "POSTN": "PDR-biased expression; efficacy still requires intervention testing.",
        "CTHRC1": "PDR-biased and endothelial expression requires safety review.",
        "SULF1": "Cross-disease balanced but ocular mechanism remains unvalidated.",
        "FN1": "Broad ECM and non-target expression limits specificity.",
        "THBS2": "Fails CPM >= 2 in all patients and is strongly PDR-biased.",
        "AEBP1": "PDR-biased and lacks direct ocular intervention evidence.",
        "IGFBP7": "Very broad endothelial expression.",
        "CFH": "Retinal-protective factor; do not infer inhibition benefit.",
        "TIMP1": "Context-dependent and broadly expressed.",
        "SERPINE1": "Context-dependent despite strong expression support.",
    })
    order = {"Primary_candidate": 0, "Mechanistic_candidate": 1, "Emerging_candidate": 2, "Risk_review": 3, "Do_not_inhibit": 4}
    result["_order"] = result["integrated_decision"].map(order).fillna(5)
    return result.sort_values(["_order", "gene"]).drop(columns="_order")


def write_documents(claims, targets, core):
    accepted = claims[claims["evidence_level"] == "Accepted"]
    boundary = claims[claims["evidence_level"] == "Supported_with_boundary"]
    rejected = claims[claims["evidence_level"] == "Exploratory_or_not_accepted"]
    summary = [
        "# ", "",
        "## ", "",
        "PVR  PDR ,  ECM/TGF-beta ., ,  VEGF .",
        "",
        "## ", "",
    ]
    summary += [f"- {row.claim} : {row.evidence} : {row.boundary}" for row in accepted.itertuples()]
    summary += ["", "## ", ""]
    summary += [f"- {row.claim} : {row.evidence} : {row.boundary}" for row in boundary.itertuples()]
    summary += ["", "## ", ""]
    summary += [f"- {row.claim} : {row.evidence} : {row.boundary}" for row in rejected.itertuples()]
    summary += [
        "", "## ", "",
        "- `POSTN`: , ,  PDR.",
        "- `CTHRC1`: , , .",
        "- `SULF1`: , .",
        "- `CFH`: ; `TIMP1`  `SERPINE1`: .",
        "- : `FN1`, `IGFBP7`, `TIMP1` ; , .",
        "-  PVR :  25  8/8  fibroblast , ; ,  RUNX1 .",
    ]
    (OUT / "mechanism_evidence_summary.md").write_text("\n".join(summary), encoding="utf-8")

    boundaries = ["# ", ""]
    boundaries += [f"- **{row.claim}**: {row.boundary}" for row in claims.itertuples()]
    boundaries += [
        "- ****: , .",
        "- ****: , .",
        "- ** PVR **:  PVR fibroblast ,  PVR/PDR , .",
        "- ****: .",
        "- ****: , .",
        "- ****: ;  PVR  FDR , .",
    ]
    (OUT / "formal_conclusion_boundaries.md").write_text("\n".join(boundaries), encoding="utf-8")

    manuscript = [
        "# ", "",
        "## 1.  PVR/PDR ", "",
        ",  45,205 .PVR  RPE , Müller ,  PDR , , .",
        "",
        "## 2. ", "",
        " 10,699 ,  Shared_ECM_TGFB_high . 2,335 ,  6/6  PVR  9/9  PDR .PVR  Müller  RPE , PDR ,  ECM/TGF-beta .",
        "",
        "## 3. ", "",
        f" pseudobulk  {len(core)}  15/15  CPM ≥ 1. 25/25;  CPM ≥ 2  15/15  THBS2 . top 20%–30% ECM/TGF-beta , .",
        "",
        "## 4. ", "",
        "25  PVR/PDR , ., , .",
        "",
        "## 5.  PVR ", "",
        " SCP2582  PVR ,  25  fibroblast  PVR , 8/8 ,  1.561, .,  RUNX1 .",
        "",
        "## 6.  VEGF ", "",
        "`POSTN` , 15/15 , , , ,  PDR.`CTHRC1` , `SULF1` .`CFH` , `TIMP1`  `SERPINE1` .",
        "",
        "## 7. ", "",
        " PVR/PDR ,  GEO , ., , .",
        "",
        "## 8. ", "",
        "`FN1`, `IGFBP7`  `TIMP1` . PXD077831  RRD ,  PVR  `FN1`  `TIMP1` ,  FDR, , .",
    ]
    (OUT / "manuscript_results_draft.md").write_text("\n".join(manuscript), encoding="utf-8")

    figure_plan = [
        "# ", "",
        "##  1: , ",
        "- ; ;  UMAP .",
        "##  2: ",
        "-  UMAP; ; /; .",
        "##  3: ",
        "- ; 25 ; ; /.",
        "##  4:  VEGF ",
        "- ; ; ; .",
        "##  5: ",
        "-  `mechanism_evidence_model.png` , , .",
        "## ",
        "- , ,  pseudobulk , , , ,  PVR .",
    ]
    (OUT / "manuscript_figure_plan.md").write_text("\n".join(figure_plan), encoding="utf-8")


def box(ax, xy, width, height, text, color, fontsize=10):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#333333", linewidth=1.2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, start, end, color="#555555"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=color))


def plot_mechanism_model():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Integrated evidence model: source-associated divergence and shared fibrotic convergence", fontsize=16, weight="bold")

    box(ax, (0.5, 5.6), 3.2, 1.2, "PVR-associated states\nRPE inflammatory\nMuller glial stress", "#b9d9f3", 11)
    box(ax, (0.5, 2.8), 3.2, 1.2, "PDR-associated states\nPericyte contractile\nVascular-wall features", "#f4c7a1", 11)
    box(ax, (5.2, 4.15), 4.0, 1.5, "Shared_ECM_TGFB_high\n2,335 cells\n6/6 PVR + 9/9 PDR patients", "#cce8c5", 12)
    arrow(ax, (3.7, 6.2), (5.2, 5.2))
    arrow(ax, (3.7, 3.4), (5.2, 4.55))

    box(ax, (10.2, 6.2), 4.0, 1.0, "Patient-level core\n25 genes in 15/15 patients", "#e2d5f2", 10)
    box(ax, (10.2, 4.7), 4.0, 1.0, "Robustness\n25/25 after patient/cohort exclusion", "#e2d5f2", 10)
    box(ax, (10.2, 3.2), 4.0, 1.0, "External validation\nPositive in 4/4 tissue comparisons", "#e2d5f2", 10)
    arrow(ax, (9.2, 4.9), (10.2, 6.7))
    arrow(ax, (9.2, 4.9), (10.2, 5.2))
    arrow(ax, (9.2, 4.9), (10.2, 3.7))

    box(ax, (4.0, 0.55), 2.8, 1.0, "POSTN\nPrimary candidate\nPDR-biased", "#f4df75", 10)
    box(ax, (7.0, 0.55), 2.8, 1.0, "CTHRC1\nMechanistic candidate\nEndothelial risk", "#f4df75", 10)
    box(ax, (10.0, 0.55), 2.8, 1.0, "SULF1\nEmerging candidate\nCross-disease balanced", "#f4df75", 10)
    arrow(ax, (7.2, 4.15), (5.4, 1.55))
    arrow(ax, (7.2, 4.15), (8.4, 1.55))
    arrow(ax, (7.2, 4.15), (11.4, 1.55))
    ax.text(0.5, 0.65, "Boundaries:\nNo lineage tracing\nNo accepted disease-specific genes\nBulk validation is not cell-state-specific\nExpression is not intervention efficacy",
            fontsize=9, va="bottom", color="#444444")
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_evidence_model.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    claims = build_claim_matrix()
    core = build_core_gene_evidence()
    targets = build_target_evidence()
    claims.to_csv(OUT / "formal_claim_evidence_matrix.csv", index=False)
    core.to_csv(OUT / "core_gene_integrated_evidence.csv", index=False)
    targets.to_csv(OUT / "target_integrated_evidence.csv", index=False)
    write_documents(claims, targets, core)
    plot_mechanism_model()


if __name__ == "__main__":
    main()
