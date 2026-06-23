# PVR-PDR shared fibrotic-state analysis

This repository is a public release package for the manuscript:

**Single-cell analysis reveals a conserved ECM/TGF-beta fibrotic signaling state across human proliferative vitreoretinal diseases**

Repository URL: https://github.com/clexa0117/pvr-pdr-shared-fibrotic-state

Zenodo archived release DOI for `v1.0.0`: https://doi.org/10.5281/zenodo.20816557

The package contains a compact English-only public release set for manuscript traceability: analysis scripts, selected derived evidence tables, reproducibility environment records, source-data accession records, and release metadata. Large raw datasets, intermediate count matrices, per-cell annotation tables, QC/UMAP image panels, generated figure images, full manuscript drafts, author-contact details, and submission-only files are not included.

## Repository contents

- `scripts/`: public formal analysis scripts for reproducing the selected derived evidence tables.
- `manifests/`: source dataset inventories.
- `ccs_signaling_analysis/`: CCS-oriented signaling-axis, TF target-program, receptor-availability, and POSTN ECM-cell interaction summaries.
- `ccs_communication_boundary/`: isolated OmniPath-resource ligand-receptor screen used to test and limit shared-state-specific communication claims.
- `results/`: selected key formal result tables, summaries, and reproducibility manifests.
- `public_release_manifest.csv`: file-level manifest for release files other than the manifest itself, using English package paths and source aliases.
- `DATA_AVAILABILITY.md`: public accession list and boundaries for data reuse.
- `CODE_AVAILABILITY.md`: reproducibility entry points and environment notes.
- `.zenodo.json` and `CITATION.cff`: metadata records for archiving and citation.

## Data availability

Primary and validation data are public and should be downloaded from their original repositories:

- GEO: `GSE294329`, `GSE245561`, `GSE165784`, `GSE179603`, `GSE60436`, `GSE41019`, `GSE228934`, `GSE274480`
- Single Cell Portal: `SCP2582`
- ProteomeXchange/PRIDE: `PXD077831`, `PXD070155`

Raw archives, raw count matrices, downloaded portal objects, and large filtered matrices are excluded from this repository to keep the release lightweight and to preserve provenance through the original accession records.

## Reproducibility

The intended workflow is documented in:

- `public_release_manifest.csv`
- `results/reproducibility_freeze/requirements_frozen.txt`
- `results/reproducibility_freeze/environment_manifest.csv`

Some scripts require local copies of the public data under `data/`, matching the structure described by the manifests. This release prioritizes transparent code and derived evidence tables rather than redistribution of raw public data.

The repository therefore has two intended uses. First, readers can inspect the selected derived tables, manifests, and reproducibility records directly. Second, readers can rerun the scripts after preparing the required public raw or intermediate inputs locally. Inputs not redistributed here include `data/raw/`, `data/validation/`, `results/formal_qc/filtered_matrices/`, and local integration folders that record larger exploratory or legacy source analyses, such as `legacy_archive/` and `public_database_upgrade_exploration/`.

Full manuscript drafts, submission-only Word/PDF artifacts, generated figure images, figure-only manifests, per-cell annotation exports, all-protein differential tables, full gene-level pseudobulk tables, full ligand-receptor pair score matrices, local test files, author-contact details, cached external resources, and manuscript-generation helper scripts are intentionally excluded from the public release.

## Citation

Please cite the manuscript when available. The archived public release DOI for this `v1.0.0` initial public release is https://doi.org/10.5281/zenodo.20816557.

## License

This repository uses a mixed academic open-release license:

- Code in `scripts/` and other executable source files is released under the MIT License.
- Derived tables, manifests, source-data accession notes, and documentation are released under Creative Commons Attribution 4.0 International (CC BY 4.0).
- Public source datasets are not redistributed here and remain governed by the terms of their original repositories.
