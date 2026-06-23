# CCS signaling analysis

This folder contains the compact public materials for the Cell Communication and Signaling-oriented signaling analysis.

The analysis evaluates the signaling context of the shared fibrotic state using fixed pathway scores, TF target-program scores, receptor-availability panels, curated ligand-receptor expression axes, and a POSTN ECM-cell interaction summary.

## Public entry points

```powershell
python ccs_signaling_analysis\scripts\run_ccs_signaling_axis_analysis.py
python ccs_signaling_analysis\scripts\run_ccs_tf_receptor_availability_analysis.py
```

The full local workflow also generated figure panels and manuscript-copy materials. Those generated images and submission-only files are intentionally excluded from this minimal public release.

## Included public result tables

- `results/pathway_activity_shared_vs_other_statistics.csv`
- `results/tf_target_program_activity_shared_vs_other_statistics.csv`
- `results/receptor_availability_activity_shared_vs_other_statistics.csv`
- `results/receptor_gene_shared_vs_other_statistics.csv`
- `results/curated_ligand_receptor_axis_shared_vs_other_statistics.csv`
- `results/curated_ligand_receptor_top_sender_summary.csv`
- `results/postn_ecm_interaction_node_summary.csv`
- gene-set coverage tables for pathways, TF programs, and receptor panels

## Interpretation boundary

These outputs support an ECM remodeling, ECM-integrin/FAK, TGF-beta/SMAD, YAP/TAZ/TEAD, and NF-kB signaling-state context. They do not prove spatial cell-cell communication, receptor activation, ligand binding, causality, or therapeutic efficacy.
