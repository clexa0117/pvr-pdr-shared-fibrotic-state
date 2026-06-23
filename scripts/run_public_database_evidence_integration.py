from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "legacy_archive" / "2026-06-14_" / "additional_public_database_exploration"
OUT = ROOT / "results" / "public_database_validation"

SELECTED_OUTPUTS = [
    "results/cross_cohort_protein_validation/candidate_cross_cohort_detection_matrix.csv",
    "results/cross_cohort_protein_validation/candidate_cross_cohort_detection_details.csv",
    "results/cross_cohort_protein_validation/figure_candidate_cross_cohort_detection.png",
    "results/PXD077831_direct_PVR/all_protein_direct_pvr_differential.csv",
    "results/PXD077831_direct_PVR/candidate_direct_pvr_validation.csv",
    "results/PXD077831_direct_PVR/candidate_sample_matrix.csv",
    "results/PXD077831_direct_PVR/figure_candidate_baseline_vitreous_heatmap.png",
    "results/PXD077831_direct_PVR/figure_direct_pvr_volcano.png",
    "results/PXD070155_recurrence_proteomics/all_protein_differential.csv",
    "results/PXD070155_recurrence_proteomics/candidate_protein_validation.csv",
    "results/PXD070155_recurrence_proteomics/figure_candidate_protein_heatmap.png",
    "results/PXD070155_recurrence_proteomics/figure_recurrence_proteomics_volcano.png",
    "results/translational_evidence/candidate_translational_evidence_matrix.csv",
    "results/translational_evidence/figure_translational_evidence_matrix.png",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_key_results(cross: pd.DataFrame, direct: pd.DataFrame, direct_all: pd.DataFrame) -> None:
    four = set(cross.loc[cross["independent_proteomics_cohort_count"] == 4, "gene"])
    three = set(cross.loc[cross["independent_proteomics_cohort_count"] == 3, "gene"])
    if four != {"FN1", "IGFBP7", "TIMP1"}:
        raise ValueError(f"Unexpected four-project detection set: {sorted(four)}")
    if three != {"CFH", "SERPINE1"}:
        raise ValueError(f"Unexpected three-project detection set: {sorted(three)}")
    if int(direct["detected"].sum()) != 5:
        raise ValueError("Expected five detected prespecified candidates in PXD077831")
    if int((direct_all["exact_permutation_fdr"] < 0.05).sum()) != 0:
        raise ValueError("PXD077831 must not be integrated as having FDR-significant proteins")


def copy_selected_outputs() -> pd.DataFrame:
    rows = []
    for relative in SELECTED_OUTPUTS:
        source = SOURCE / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = OUT / Path(relative).relative_to("results")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append(
            {
                "source_path": source.relative_to(ROOT).as_posix(),
                "formal_path": destination.relative_to(ROOT).as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "integration_role": "Selected formal public-database evidence",
            }
        )
    return pd.DataFrame(rows)


def write_summary(cross: pd.DataFrame, direct: pd.DataFrame, direct_all: pd.DataFrame) -> None:
    fn1 = direct.loc[direct["gene"] == "FN1"].iloc[0]
    timp1 = direct.loc[direct["gene"] == "TIMP1"].iloc[0]
    lines = [
        "# ",
        "",
        "## ",
        "",
        "- .",
        "- `FN1`, `IGFBP7`, `TIMP1` ; `CFH`, `SERPINE1` .",
        "- `PXD077831`  RRD ,  PVR  8  PVR  8 , .",
        f"- `PXD077831`  5/10 ; `FN1`  `TIMP1` (Hedges g = {fn1['hedges_g']:.3f}, {timp1['hedges_g']:.3f}),  p = {fn1['exact_permutation_p']:.3f}, {timp1['exact_permutation_p']:.3f}.",
        f"-  {int((direct_all['exact_permutation_p'] < 0.05).sum())} ,  FDR.",
        "",
        "## ",
        "",
        "- , .",
        "- `PXD077831`  PVR , .",
        "- `PXD070155`  DRI/DRII , .",
        "- , .",
        "",
        "## ",
        "",
        "-  `scripts/run_public_database_evidence_integration.py` .",
        "- `integration_manifest.csv` , ,  SHA-256.",
    ]
    (OUT / "public_database_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = copy_selected_outputs()
    cross = pd.read_csv(OUT / "cross_cohort_protein_validation/candidate_cross_cohort_detection_matrix.csv")
    direct = pd.read_csv(OUT / "PXD077831_direct_PVR/candidate_direct_pvr_validation.csv")
    direct_all = pd.read_csv(OUT / "PXD077831_direct_PVR/all_protein_direct_pvr_differential.csv")
    validate_key_results(cross, direct, direct_all)
    manifest.to_csv(OUT / "integration_manifest.csv", index=False)
    write_summary(cross, direct, direct_all)
    print(f"Public-database evidence integration written to {OUT}")


if __name__ == "__main__":
    main()
