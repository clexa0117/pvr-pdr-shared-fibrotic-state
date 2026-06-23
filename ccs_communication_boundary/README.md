# CCS formal communication isolation

This isolated folder tests whether the current PVR-PDR shared fibrotic signaling-state result can be strengthened by a formal cell-cell communication layer.

## Status

Completed on 2026-06-22. The result does **not** support adding a strong shared-state-specific communication claim to the manuscript.

The safest interpretation is:

> Formal ligand-receptor inference did not support a strong shared-state-specific communication claim. It supports ECM/integrin and TGF-beta-related signaling-program context, while keeping physical cell-cell communication, spatial proximity, receptor activation, and causal signaling as unproven.

## Why This Is Isolated

The current manuscript already contains curated ligand-receptor expression context. This folder asks a stricter question: do prespecified ligand-receptor axes show patient-level enrichment into `Shared_ECM_TGFB_high` compared with other fibrotic-related states?

The answer is no under the conservative rule used here.

## Main Entry Point

- `scripts/run_formal_communication_isolation.py`: downloads and caches the OmniPath ligand-receptor resource at runtime if needed, builds prespecified LR prior pairs, scores sender-receiver expression, runs patient-level shared-vs-other statistics, adds expression-matched random ligand controls, creates a NicheNet-style target-program interpretation table, and writes perturbation/literature boundary tables.

## Key Outputs

- `results/formal_communication_summary.md`: readable summary and bottom-line interpretation.
- `results/formal_lr_prior_pairs.csv`: prespecified LR pairs with OmniPath support annotation.
- `results/formal_lr_pair_scores.csv.gz`: full sender cell type, receiver state, ligand, receptor, and score table; this large intermediate table is not included in the minimal public release and can be regenerated from the script.
- `results/formal_lr_axis_shared_vs_other_statistics.csv`: patient-level shared-vs-other statistics for each LR axis.
- `results/formal_lr_random_ligand_control.csv`: expression-matched random ligand control.
- `results/nichenet_style_ligand_target_explanation.csv`: NicheNet-style target-program interpretation, not native NicheNet output.
- `results/model_perturbation_support_boundary.csv`: model-system perturbation and induction support boundary table.
- `results/literature_spatial_immuno_support.csv`: literature-level tissue/spatial/immuno support table.
- `results/package_feasibility.csv`: local native package availability record for LIANA, CellPhoneDB, CellChat, NicheNet, and related tools.
- `results/input_manifest.csv`: input traceability manifest. The minimal public release uses the root `public_release_manifest.csv` as its package-level file manifest.

The cached OmniPath resource file is not included in this minimal public release. It is recreated under `resources/` when the script is rerun with network access, or it can be supplied manually from OmniPath before rerunning.

## Main Result

- LR prior: 83 pairs across 9 prespecified axes.
- OmniPath-supported pairs: 57/83.
- Axes meeting conservative shared-state-specific communication rule: 0/9.
- Axes with boundary or negative direction: 9/9.
- Random ligand control: no real axis exceeded the expression-matched random-ligand benchmark.

## Interpretation Boundary

This folder should not be used to claim CellChat-like, CellPhoneDB-like, LIANA-native, NicheNet-native, spatial, or causal communication evidence.

It can support a cautious sentence such as:

> An isolated OmniPath-resource ligand-receptor analysis with LIANA-style consensus ranking did not identify a robust shared-state-specific communication axis; therefore, the shared state is interpreted primarily as a convergent ECM/TGF-beta signaling-state context rather than a proven physical communication event.
