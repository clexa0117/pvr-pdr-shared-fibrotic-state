from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public_database_upgrade_exploration" / "results" / "scp2582_independent_pvr_validation"
OUT = ROOT / "results" / "independent_human_pvr_validation"
FINAL_OUT = ROOT / "final_submission_materials" / "03_" / "independent_human_pvr_validation"

SELECTED = [
    "summary.md",
    "gene_level_summary.md",
    "module_gene_coverage.csv",
    "donor_fibroblast_vs_other_effects.csv",
    "fibroblast_vs_other_effect_summary.csv",
    "matched_random_benchmark.csv",
    "celltype_donor_balanced_summary.csv",
    "gene_fibroblast_enrichment_summary.csv",
    "candidate_gene_fibroblast_enrichment_summary.csv",
    "donor_fibroblast_vs_other_module_effects.png",
    "celltype_core_score_donor_balanced.png",
    "matched_random_fibroblast_effect_benchmark.png",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_selected(destination_root: Path) -> pd.DataFrame:
    destination_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in SELECTED:
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError(source)
        destination = destination_root / name
        shutil.copy2(source, destination)
        rows.append({
            "source_path": source.relative_to(ROOT).as_posix(),
            "formal_path": destination.relative_to(ROOT).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "integration_role": "Independent human PVR single-cell validation",
        })
    return pd.DataFrame(rows)


def validate_results(out_dir: Path) -> None:
    effect = pd.read_csv(out_dir / "fibroblast_vs_other_effect_summary.csv").iloc[0]
    random = pd.read_csv(out_dir / "matched_random_benchmark.csv").iloc[0]
    genes = pd.read_csv(out_dir / "gene_fibroblast_enrichment_summary.csv")
    core_genes = genes[genes["is_core_25"] == True]  # noqa: E712
    if int(effect["n_donors"]) != 8:
        raise ValueError("Expected 8 SCP2582 donors")
    if int(effect["donors_with_fibroblast_higher_core"]) != 8:
        raise ValueError("Expected fixed core to be positive in 8/8 donors")
    if float(random["empirical_upper_p"]) > 0.0011:
        raise ValueError("Expected matched random empirical p <= 0.0011")
    if len(core_genes) != 25:
        raise ValueError("Expected 25 represented core genes")
    if int((core_genes["positive_mean_effect_donors"] >= 7).sum()) != 25:
        raise ValueError("Expected 25/25 core genes positive in at least 7/8 donors")


def write_summary(out_dir: Path, final_dir: Path, manifest: pd.DataFrame) -> None:
    effect = pd.read_csv(out_dir / "fibroblast_vs_other_effect_summary.csv").iloc[0]
    random = pd.read_csv(out_dir / "matched_random_benchmark.csv").iloc[0]
    genes = pd.read_csv(out_dir / "gene_fibroblast_enrichment_summary.csv")
    postn = genes.loc[genes["gene"] == "POSTN"].iloc[0]
    cthrc1 = genes.loc[genes["gene"] == "CTHRC1"].iloc[0]
    sulf1 = genes.loc[genes["gene"] == "SULF1"].iloc[0]
    lines = [
        "#  PVR ",
        "",
        "## ",
        "",
        "- `SCP2582`  Seurat  6,372 , 8  PVR .",
        "-  25  25/25 , .",
        f"- fibroblast  PVR  {effect['core_effect_mean']:.3f}, 95% CI {effect['core_effect_lower_95']:.3f}-{effect['core_effect_upper_95']:.3f}, 8/8 .",
        f"-  {effect['generic_effect_mean']:.3f}; ,  {effect['residual_effect_mean']:.3f}.",
        f"- ,  {random['percentile']:.3f} ,  p = {random['empirical_upper_p']:.4f}.",
        "- 25/25  7/8  fibroblast .",
        f"- POSTN, CTHRC1  SULF1  8/8 ; fibroblast  {postn['fibroblast_detection_mean']:.3f}, {cthrc1['fibroblast_detection_mean']:.3f}  {sulf1['fibroblast_detection_mean']:.3f}.",
        "",
        "## ",
        "",
        "-  PVR , .",
        "- ,  FASTQ/.",
        "-  RUNX1 , .",
        "-  PVR fibroblast ,  PVR/PDR .",
        "",
        "## ",
        "",
        "-  `scripts/run_independent_human_pvr_validation_integration.py` .",
        "- `integration_manifest.csv` , ,  SHA-256.",
    ]
    (out_dir / "independent_human_pvr_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(out_dir / "independent_human_pvr_validation_summary.md", final_dir / "independent_human_pvr_validation_summary.md")
    manifest.to_csv(out_dir / "integration_manifest.csv", index=False)
    shutil.copy2(out_dir / "integration_manifest.csv", final_dir / "integration_manifest.csv")


def main() -> None:
    manifest = copy_selected(OUT)
    copy_selected(FINAL_OUT)
    validate_results(OUT)
    write_summary(OUT, FINAL_OUT, manifest)
    print(f"Independent human PVR validation integrated into {OUT}")


if __name__ == "__main__":
    main()
