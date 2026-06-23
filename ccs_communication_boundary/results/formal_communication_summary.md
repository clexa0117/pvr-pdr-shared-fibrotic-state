# Formal communication isolation summary

## Bottom line

- LR prior pairs tested: 83 across 9 prespecified axes.
- OmniPath-supported prior pairs: 57/83.
- Axes meeting the conservative shared-state-specific communication rule: 0/9.
- Axes with boundary/negative direction for shared-state-specific communication: 9/9.
- This folder should be treated as isolated exploratory support. It does not replace the current manuscript claim unless reviewed and promoted.

## Interpretation rule

A formal communication claim requires patient-level shared-state enrichment, not merely ligand expression in a sender cell type and receptor expression in the shared state. If an axis fails this rule, the safest wording is that the shared state is better supported as a signaling-state convergence than as a proven physical cell-cell communication event.

## Axis-level formal LR statistics

| feature | patients | positive_patients | mean_difference_shared_minus_other | bootstrap_ci_low | bootstrap_ci_high | wilcoxon_fdr | main_text_claim_status |
|:---------------------|-----------:|--------------------:|-------------------------------------:|-------------------:|--------------------:|---------------:|:-------------------------------------------------------------|
| CCL_CCR | 15 | 3 | -0.102196 | -0.224051 | 0.016496 | 0.108488 | boundary_or_negative_for_shared_state_specific_communication |
| CXCL_CXCR | 15 | 2 | -0.0974904 | -0.20851 | 0.0129614 | 0.0334201 | boundary_or_negative_for_shared_state_specific_communication |
| FN1_integrin | 15 | 4 | -0.00476841 | -0.00947383 | -0.000118742 | 0.128394 | boundary_or_negative_for_shared_state_specific_communication |
| PDGF_PDGFR | 15 | 4 | -0.0222026 | -0.0350311 | -0.0102165 | 0.0229472 | boundary_or_negative_for_shared_state_specific_communication |
| POSTN_integrin | 15 | 4 | -0.00754128 | -0.0127444 | -0.00271432 | 0.0230713 | boundary_or_negative_for_shared_state_specific_communication |
| SPP1_CD44_integrin | 15 | 2 | -0.00391879 | -0.00615083 | -0.00182764 | 0.0157471 | boundary_or_negative_for_shared_state_specific_communication |
| TGFB_TGFBR | 15 | 7 | -0.00087104 | -0.0131322 | 0.0107533 | 0.934082 | boundary_or_negative_for_shared_state_specific_communication |
| THBS_TGFB_activation | 15 | 4 | -0.00423025 | -0.00754333 | -0.00144034 | 0.0334201 | boundary_or_negative_for_shared_state_specific_communication |
| VEGF_VEGFR | 15 | 2 | -0.0190959 | -0.0265871 | -0.0110329 | 0.00384521 | boundary_or_negative_for_shared_state_specific_communication |

## Random ligand control

| axis | actual_mean_difference_product_detection_score | random_iterations | random_mean | random_ci_low | random_ci_high | empirical_upper_tail_p | actual_percentile_vs_random | control_interpretation |
|:---------------------|-------------------------------------------------:|--------------------:|--------------:|----------------:|-----------------:|-------------------------:|------------------------------:|:-------------------------------------------|
| CCL_CCR | -0.0123058 | 80 | -0.0123063 | -0.0245956 | -0.00580268 | 0.617284 | 0.39375 | actual_not_above_expression_matched_random |
| CXCL_CXCR | -0.268561 | 80 | -0.321332 | -0.72879 | -0.142136 | 0.45679 | 0.55625 | actual_not_above_expression_matched_random |
| FN1_integrin | -0.79771 | 80 | -0.436182 | -1.37154 | -0.127551 | 0.91358 | 0.09375 | actual_not_above_expression_matched_random |
| PDGF_PDGFR | -0.205495 | 80 | -0.186277 | -0.352569 | -0.0911814 | 0.679012 | 0.33125 | actual_not_above_expression_matched_random |
| POSTN_integrin | -0.347967 | 80 | -0.492431 | -1.65396 | -0.159658 | 0.493827 | 0.51875 | actual_not_above_expression_matched_random |
| SPP1_CD44_integrin | -1.0263 | 80 | -0.335191 | -0.984861 | -0.110273 | 0.975309 | 0.03125 | actual_not_above_expression_matched_random |
| TGFB_TGFBR | -0.0226475 | 80 | -0.0206797 | -0.0507786 | -0.00604292 | 0.728395 | 0.28125 | actual_not_above_expression_matched_random |
| THBS_TGFB_activation | -0.374581 | 80 | -0.478731 | -1.53069 | -0.132237 | 0.506173 | 0.50625 | actual_not_above_expression_matched_random |
| VEGF_VEGFR | -1.32659 | 80 | -0.937208 | -2.14696 | -0.398631 | 0.851852 | 0.15625 | actual_not_above_expression_matched_random |

## NicheNet-style target-program interpretation

| axis | ligand | shared_state_samples_with_gated_pair | median_shared_consensus_score | most_common_top_sender | supported_target_programs | interpretation |
|:---------------------|:---------|---------------------------------------:|--------------------------------:|:-------------------------|:------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| FN1_integrin | FN1 | 15 | 0.968809 | Fibroblast_Mesenchymal | ECM_integrin_FAK;ECM_remodeling;TEAD_YAP_TAZ_targets;YAP_TAZ_TEAD | target-program context supported, but formal LR score is not shared-state specific |
| POSTN_integrin | POSTN | 15 | 0.966667 | Fibroblast_Mesenchymal | ECM_integrin_FAK;ECM_remodeling;TEAD_YAP_TAZ_targets;YAP_TAZ_TEAD | target-program context supported, but formal LR score is not shared-state specific |
| SPP1_CD44_integrin | SPP1 | 15 | 0.97012 | Myeloid | ECM_integrin_FAK;NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| THBS_TGFB_activation | THBS2 | 11 | 0.969639 | Fibroblast_Mesenchymal | ECM_remodeling;SMAD2_3_TGFbeta_targets;TGF_beta_SMAD | target-program context supported, but formal LR score is not shared-state specific |
| THBS_TGFB_activation | THBS1 | 15 | 0.963454 | Fibroblast_Mesenchymal | ECM_remodeling;SMAD2_3_TGFbeta_targets;TGF_beta_SMAD | target-program context supported, but formal LR score is not shared-state specific |
| TGFB_TGFBR | TGFB2 | 14 | 0.911807 | Fibroblast_Mesenchymal | SMAD2_3_TGFbeta_targets;TGF_beta_SMAD | target-program context supported, but formal LR score is not shared-state specific |
| TGFB_TGFBR | TGFB3 | 14 | 0.896225 | Fibroblast_Mesenchymal | SMAD2_3_TGFbeta_targets;TGF_beta_SMAD | target-program context supported, but formal LR score is not shared-state specific |
| TGFB_TGFBR | TGFB1 | 15 | 0.892771 | T_NK | SMAD2_3_TGFbeta_targets;TGF_beta_SMAD | target-program context supported, but formal LR score is not shared-state specific |
| CXCL_CXCR | CXCL8 | 9 | 0.737349 | Myeloid | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CXCL_CXCR | CXCL12 | 9 | 0.730442 | Fibroblast_Mesenchymal | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CXCL_CXCR | CXCL3 | 9 | 0.720964 | Myeloid | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CXCL_CXCR | CXCL2 | 9 | 0.71743 | Myeloid | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CXCL_CXCR | CXCL1 | 7 | 0.667135 | Myeloid | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CCL_CCR | CCL5 | 3 | 0.630857 | T_NK | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CCL_CCR | CCL3 | 3 | 0.607229 | Myeloid | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CCL_CCR | CCL4 | 3 | 0.594444 | Myeloid | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| CCL_CCR | CCL2 | 3 | 0.542503 | Myeloid | NFkB_inflammatory;RELA_NFKB_targets | target-program context supported, but formal LR score is not shared-state specific |
| VEGF_VEGFR | VEGFA | 15 | 0.944578 | Myeloid | | weak target-program and LR support; keep as boundary or do not emphasize |
| VEGF_VEGFR | PGF | 12 | 0.936145 | Fibroblast_Mesenchymal | | weak target-program and LR support; keep as boundary or do not emphasize |
| VEGF_VEGFR | VEGFB | 14 | 0.923561 | Myeloid | | weak target-program and LR support; keep as boundary or do not emphasize |
| PDGF_PDGFR | PDGFB | 15 | 0.917001 | Endothelial | | weak target-program and LR support; keep as boundary or do not emphasize |
| PDGF_PDGFR | PDGFC | 15 | 0.909772 | Fibroblast_Mesenchymal | | weak target-program and LR support; keep as boundary or do not emphasize |
| PDGF_PDGFR | PDGFA | 15 | 0.908166 | Fibroblast_Mesenchymal | | weak target-program and LR support; keep as boundary or do not emphasize |
| PDGF_PDGFR | PDGFD | 13 | 0.858554 | Endothelial | | weak target-program and LR support; keep as boundary or do not emphasize |

## Model perturbation support boundary

- Perturbation/model support rows collected: 27.
- Treat these rows as model-system directional support only, not therapeutic validation.

## Literature-level spatial/immuno support boundary

- Literature support rows collected: 8.
- These rows are useful for a supplementary evidence table or Discussion sentence, but they are not generated by this study.
