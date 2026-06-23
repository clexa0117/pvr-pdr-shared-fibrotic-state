import csv
import gzip
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
RESULTS = ROOT / "results"


def count_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def dimensions(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("%"):
                return tuple(map(int, line.split()))
    raise ValueError(path)


def validate_10x(accession):
    rows = []
    folder = RAW / accession / "files"
    for matrix in sorted(folder.glob("*_matrix.mtx.gz")):
        prefix = matrix.name.removesuffix("_matrix.mtx.gz")
        genes, cells, entries = dimensions(matrix)
        barcodes = count_lines(folder / f"{prefix}_barcodes.tsv.gz")
        features = count_lines(folder / f"{prefix}_features.tsv.gz")
        rows.append([accession, prefix, "10x_mtx", genes, cells, entries, barcodes, features, genes == features and cells == barcodes])
    return rows


def validate_gse165784():
    rows = []
    for path in sorted((RAW / "GSE165784/files").glob("*")):
        if path.name.endswith("_matrix.tsv.gz"):
            rows.append(["GSE165784", path.name.removesuffix("_matrix.tsv.gz"), "dense_tsv", count_lines(path) - 1, "", "", "", "", True])
        elif path.name.endswith("_matrix_10X.tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                names = [member.name for member in archive.getmembers() if member.isfile()]
            valid = any("barcodes.tsv" in n for n in names) and any(("genes.tsv" in n or "features.tsv" in n) for n in names) and any("matrix.mtx" in n for n in names)
            rows.append(["GSE165784", path.name.removesuffix("_matrix_10X.tar.gz"), "nested_10x_tar", "", "", "", "", "", valid])
    return rows


def main():
    rows = validate_10x("GSE294329") + validate_10x("GSE245561") + validate_gse165784()
    RESULTS.mkdir(exist_ok=True)
    header = ["accession", "sample", "format", "genes", "cells", "entries", "barcodes", "features", "valid"]
    with (RESULTS / "core_download_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    totals = {acc: sum(int(row[4]) for row in rows if row[0] == acc and row[4] != "") for acc in ("GSE294329", "GSE245561")}
    lines = [
        "# Core single-cell download validation", "",
        f"- Files checked: {len(rows)} samples",
        f"- All files valid: {all(row[-1] for row in rows)}",
        f"- GSE294329 total matrix columns/cells: {totals['GSE294329']}",
        f"- GSE245561 total matrix columns/cells: {totals['GSE245561']}", "",
        "| Accession | Sample | Format | Cells | Valid |",
        "|---|---|---|---:|---|",
    ]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[4] or 'n/a'} | {r[-1]} |" for r in rows]
    (RESULTS / "core_download_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all(row[-1] for row in rows):
        raise SystemExit("Validation failure")


if __name__ == "__main__":
    main()
