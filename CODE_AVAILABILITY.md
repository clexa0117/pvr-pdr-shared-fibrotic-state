# Code availability

Formal analysis scripts are provided in `scripts/`. Publicly relevant entry points are listed below; the full local frozen workflow also included manuscript assembly steps and is retained in the project workspace rather than this public release.

This public package intentionally excludes manuscript-draft and journal-submission helper scripts that were used only to assemble local Word/PDF submission files.

The scripts are provided for transparency and rerun after data preparation; this lightweight release is not a complete raw-data mirror. Several entry points require public inputs or intermediate matrices to be restored locally according to `DATA_AVAILABILITY.md`, `public_release_manifest.csv`, and `manifests/source_dataset_inventory.csv`. In particular, raw GEO matrices should be prepared under `data/raw/`, validation expression data under `data/validation/`, filtered count matrices under `results/formal_qc/filtered_matrices/`, and larger legacy or public-database integration sources under their documented local source folders. The repository itself includes selected derived result tables and reproducibility records, not the full raw or intermediate data tree.

Recommended entry points:

1. `scripts/validate_core_downloads.py`
2. `scripts/run_stop_go_prescreen.py`
3. `scripts/run_formal_qc.py`
4. `scripts/run_formal_clustering_annotation.py`
5. `scripts/run_fibrotic_state_analysis.py`
6. `scripts/run_patient_pseudobulk_analysis.py`
7. `scripts/run_external_validation_targets.py`
8. `scripts/run_target_cellular_specificity.py`
9. `scripts/run_robustness_sensitivity.py`
10. `scripts/run_reviewer_risk_validation.py`
11. `scripts/run_public_database_evidence_integration.py`
12. `scripts/run_independent_human_pvr_validation_integration.py`
13. `scripts/run_mechanism_evidence_integration.py`
14. `ccs_signaling_analysis/scripts/run_ccs_signaling_axis_analysis.py`
15. `ccs_signaling_analysis/scripts/run_ccs_tf_receptor_availability_analysis.py`
16. `ccs_communication_boundary/scripts/run_formal_communication_isolation.py`

The isolated CCS communication script downloads and caches the OmniPath ligand-receptor resource at runtime if it is not already present locally; the cached resource file is not included in this minimal public package.

The local environment snapshot is recorded in `results/reproducibility_freeze/environment_manifest.csv` and `results/reproducibility_freeze/requirements_frozen.txt`.
