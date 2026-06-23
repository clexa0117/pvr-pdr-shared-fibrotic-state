import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "results/stop_go_prescreen"

MODULES = {
    "fibrosis_ecm": {"COL1A1","COL1A2","COL3A1","COL5A1","COL5A2","COL6A1","COL6A2","COL6A3","COL12A1","FN1","DCN","LUM","SPARC","TGFBI","POSTN","CTHRC1"},
    "contractile": {"ACTA2","TAGLN","MYL9","TPM1","TPM2","CNN1","CALD1"},
    "tgfb_mechanosensing": {"TGFB1","TGFBR1","TGFBR2","CTGF","YAP1","WWTR1","ITGA5","ITGB1"},
    "rpe_origin": {"RPE65","BEST1","RLBP1","MITF","PMEL","TYR","TYRP1","KRT8","KRT18","KRT19","EPCAM"},
    "pericyte_origin": {"RGS5","PDGFRB","CSPG4","MCAM","DES","NOTCH3","COL4A1","COL4A2"},
    "endothelial": {"PECAM1","VWF","KDR","EMCN","CLDN5","ESAM","ENG"},
    "immune": {"PTPRC","TYROBP","LST1","AIF1","CD3D","NKG7","FCER1G"},
    "glial": {"RLBP1","GLUL","SLC1A3","GFAP","AQP4","CLU"},
    "angiogenesis": {"VEGFA","KDR","FLT1","ANGPT2","TEK","ESM1","PLVAP"},
    "inflammation": {"IL1B","TNF","CXCL8","CCL2","S100A8","S100A9","NFKBIA"},
}
G2M = defaultdict(list)
for module, genes in MODULES.items():
    for gene in genes:
        G2M[gene].append(module)


def sample_list():
    rows = []
    for acc, disease in (("GSE294329","PVR"),("GSE245561","PDR")):
        for matrix in sorted((RAW / acc / "files").glob("*_matrix.mtx.gz")):
            rows.append((acc, disease, matrix.name.removesuffix("_matrix.mtx.gz"), "mtx_gz", matrix))
    for path in sorted((RAW / "GSE165784/files").glob("*_matrix.tsv.gz")):
        disease = "PVR" if "RRD-ERM1" in path.name else "PDR"
        rows.append(("GSE165784", disease, path.name.removesuffix("_matrix.tsv.gz"), "dense", path))
    for gsm, folder in (("GSM5690478_PDR-FM-0609","GSM5690478/PDR-FM-0609_matrix_10X"),("GSM5690479_PDR-ERM-210630","GSM5690479/PDR-ERM-210630_matrix_10X")):
        rows.append(("GSE165784","PDR",gsm,"mtx_plain",RAW / "GSE165784/nested" / folder / "matrix.mtx"))
    return rows


def arrays(n):
    return {
        "total": np.zeros(n, np.int64), "genes": np.zeros(n, np.int32), "mito": np.zeros(n, np.int64),
        "counts": {m: np.zeros(n, np.int64) for m in MODULES},
        "detected": {m: np.zeros(n, np.int16) for m in MODULES},
    }


def update(a, gene, idx, vals):
    a["total"][idx] += vals
    a["genes"][idx] += 1
    if gene.startswith("MT-"):
        a["mito"][idx] += vals
    for module in G2M.get(gene, ()):
        a["counts"][module][idx] += vals
        a["detected"][module][idx] += 1


def read_mtx(fmt, matrix):
    if fmt == "mtx_gz":
        prefix = str(matrix).removesuffix("_matrix.mtx.gz")
        feature = Path(prefix + "_features.tsv.gz")
        op = gzip.open
    else:
        feature = matrix.parent / "genes.tsv"
        op = open
    genes = []
    with (gzip.open(feature, "rt", encoding="utf-8") if feature.suffix == ".gz" else feature.open(encoding="utf-8")) as h:
        for line in h:
            parts = line.rstrip().split("\t")
            genes.append(parts[1] if len(parts) > 1 else parts[0])
    with op(matrix, "rt", encoding="utf-8") as h:
        line = h.readline()
        while line.startswith("%"):
            line = h.readline()
        _, cells, _ = map(int, line.split())
        a = arrays(cells)
        for line in h:
            gi, ci, value = map(int, line.split())
            update(a, genes[gi-1], np.array([ci-1]), np.array([value]))
    return a


def read_dense(path):
    with gzip.open(path, "rt", encoding="utf-8") as h:
        cells = len(h.readline().rstrip().split("\t")) - 1
        a = arrays(cells)
        for line in h:
            parts = line.rstrip().split("\t")
            vals = np.fromiter((int(float(x)) for x in parts[1:]), dtype=np.int64, count=cells)
            idx = np.flatnonzero(vals)
            if idx.size:
                update(a, parts[0], idx, vals[idx])
    return a


def pct(n, d):
    return float(n / d * 100) if d else 0.0


def summarize(acc, disease, sample, a):
    total, genes = a["total"], a["genes"]
    mito = np.divide(a["mito"], total, out=np.zeros_like(total, dtype=float), where=total > 0) * 100
    qc = (total >= 500) & (genes >= 200) & (mito < 25)
    fib_cp = np.divide(a["counts"]["fibrosis_ecm"] * 10000, total, out=np.zeros_like(total,dtype=float), where=total>0)
    cutoff = np.quantile(fib_cp[qc], .75)
    fib = qc & (a["detected"]["fibrosis_ecm"] >= 3) & ((a["detected"]["contractile"] >= 1) | (fib_cp >= cutoff))
    row = {
        "accession":acc,"disease":disease,"sample":sample,"raw_cells":len(total),"qc_pass_cells":int(qc.sum()),
        "qc_pass_pct":pct(qc.sum(),len(total)),"median_umi_qc":float(np.median(total[qc])),
        "median_genes_qc":float(np.median(genes[qc])),"median_mito_pct_qc":float(np.median(mito[qc])),
        "fibrotic_candidate_cells":int(fib.sum()),"fibrotic_candidate_pct_of_qc":pct(fib.sum(),qc.sum()),
    }
    for module in MODULES:
        cp = np.divide(a["counts"][module]*10000,total,out=np.zeros_like(total,dtype=float),where=total>0)
        row[f"{module}_median_all_qc"] = float(np.median(cp[qc]))
        row[f"{module}_median_fibrotic"] = float(np.median(cp[fib]))
        row[f"{module}_positive_pct_qc"] = pct((a["detected"][module][qc]>=1).sum(),qc.sum())
        row[f"{module}_positive_pct_fibrotic"] = pct((a["detected"][module][fib]>=1).sum(),fib.sum())
    return row


def write(name, rows):
    with (OUT/name).open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for acc,disease,sample,fmt,path in sample_list():
        rows.append(summarize(acc,disease,sample,read_dense(path) if fmt=="dense" else read_mtx(fmt,path)))
    write("sample_qc_and_modules.csv",rows)
    cohorts=[]
    for acc,disease in sorted({(r["accession"],r["disease"]) for r in rows}):
        s=[r for r in rows if r["accession"]==acc and r["disease"]==disease]
        cohorts.append({"accession":acc,"disease":disease,"samples":len(s),"raw_cells":sum(r["raw_cells"] for r in s),"qc_pass_cells":sum(r["qc_pass_cells"] for r in s),"fibrotic_candidate_cells":sum(r["fibrotic_candidate_cells"] for r in s),"samples_with_at_least_50_fibrotic_candidates":sum(r["fibrotic_candidate_cells"]>=50 for r in s),"median_fibrotic_pct":float(np.median([r["fibrotic_candidate_pct_of_qc"] for r in s]))})
    write("cohort_summary.csv",cohorts)
    shared=[]
    for module in ("fibrosis_ecm","contractile","tgfb_mechanosensing"):
        for disease in ("PVR","PDR"):
            s=[r for r in rows if r["disease"]==disease and r["fibrotic_candidate_cells"]>=50]
            vals=[r[f"{module}_positive_pct_fibrotic"] for r in s]
            shared.append({"module":module,"disease":disease,"eligible_samples":len(s),"samples_with_module_in_at_least_50pct_fibrotic_cells":sum(v>=50 for v in vals),"median_positive_pct_in_fibrotic_cells":float(np.median(vals))})
    write("shared_module_screen.csv",shared)
    core=[m for m in ("fibrosis_ecm","contractile","tgfb_mechanosensing") if all(float(r["median_positive_pct_in_fibrotic_cells"])>=50 for r in shared if r["module"]==m)]
    lines=["# PVR-PDR stop/go prescreen","","## Data availability and basic quality","",f"- Samples processed: {len(rows)}",f"- PVR samples: {sum(r['disease']=='PVR' for r in rows)}",f"- PDR samples: {sum(r['disease']=='PDR' for r in rows)}",f"- Raw matrix cells: {sum(r['raw_cells'] for r in rows)}",f"- Cells passing simple QC: {sum(r['qc_pass_cells'] for r in rows)}",f"- Samples with at least 50 candidate fibrotic cells: {sum(r['fibrotic_candidate_cells']>=50 for r in rows)}/{len(rows)}","","## Core-story screen","",f"- Shared core modules passing the broad reproducibility screen: {core}",f"- Every sample has at least 50 candidate fibrotic cells: {all(r['fibrotic_candidate_cells']>=50 for r in rows)}","","| Accession | Disease | Sample | QC-pass cells | Candidate fibrotic cells | Candidate percent |","|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['accession']} | {r['disease']} | {r['sample']} | {r['qc_pass_cells']} | {r['fibrotic_candidate_cells']} | {r['fibrotic_candidate_pct_of_qc']:.1f}% |" for r in rows]
    (OUT/"stop_go_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
