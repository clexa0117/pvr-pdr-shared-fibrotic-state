import csv
import gzip
import itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/validation"; OUT=ROOT/"results/stop_go_prescreen"
MODULES={
"fibrosis_ecm":{"COL1A1","COL1A2","COL3A1","COL5A1","COL5A2","COL6A1","COL6A2","COL6A3","COL12A1","FN1","DCN","LUM","SPARC","TGFBI","POSTN","CTHRC1"},
"contractile":{"ACTA2","TAGLN","MYL9","TPM1","TPM2","CNN1","CALD1"},
"tgfb_mechanosensing":{"TGFB1","TGFBR1","TGFBR2","CTGF","YAP1","WWTR1","ITGA5","ITGB1"}}

def zscore(x):
    s=x.std(axis=1,keepdims=True);s[s==0]=1
    return (x-x.mean(axis=1,keepdims=True))/s

def scores(genes,samples):
    out={}
    for module,wanted in MODULES.items():
        present=sorted(wanted & genes.keys())
        out[module]=(present,dict(zip(samples,zscore(np.vstack([genes[g] for g in present])).mean(axis=0))))
    return out

def perm(a,b):
    a=np.asarray(a);b=np.asarray(b);obs=abs(a.mean()-b.mean());x=np.r_[a,b];n=len(a);diffs=[]
    for idx in itertools.combinations(range(len(x)),n):
        mask=np.zeros(len(x),bool);mask[list(idx)]=True;diffs.append(abs(x[mask].mean()-x[~mask].mean()))
    return (sum(d>=obs-1e-12 for d in diffs)+1)/(len(diffs)+1)

def annotation():
    result={}
    with gzip.open(DATA/"GPL6884/GPL6884.annot.gz","rt",encoding="utf-8") as h:
        for line in h:
            if line.startswith("ID\t"): header=line.rstrip().split("\t");break
        for row in csv.DictReader(h,fieldnames=header,delimiter="\t"):
            gene=(row.get("Gene symbol") or "").split("///")[0].strip()
            if gene: result[row["ID"]]=gene
    return result

def gse179603():
    genes={}; target=set().union(*MODULES.values())
    with gzip.open(DATA/"GSE179603/GSE179603_data.csv.gz","rt",encoding="utf-8",newline="") as h:
        r=csv.reader(h,delimiter=";");head=next(r);samples=[x.strip('"') for x in head[5:32]]
        for row in r:
            gene=row[0].strip('"')
            if gene in target:
                vals=[x.strip('"').replace(",",".") for x in row[5:32]]
                if all(x not in {"","NA"} for x in vals): genes[gene]=np.asarray(list(map(float,vals)))
    groups={"PVR":[s for s in samples if s.startswith("PVR_")],"gliosis":[s for s in samples if s.startswith("Gliose_")],"ILM":[s for s in samples if s.startswith("ILM_")]}
    return "GSE179603",samples,genes,groups,[("PVR","gliosis"),("PVR","ILM")]

def series(acc,mapper,comparisons,probe):
    genes=defaultdict(list);target=set().union(*MODULES.values());samples=[]
    with gzip.open(DATA/acc/f"{acc}_series_matrix.txt.gz","rt",encoding="utf-8") as h:
        table=False
        for line in h:
            if line.startswith("!series_matrix_table_begin"):table=True;continue
            if line.startswith("!series_matrix_table_end"):break
            if not table:continue
            p=[x.strip('"') for x in line.rstrip().split("\t")]
            if p[0]=="ID_REF":samples=p[1:];continue
            gene=probe.get(p[0])
            if gene in target:genes[gene].append(np.asarray(list(map(float,p[1:]))))
    genes={g:np.vstack(v).mean(axis=0) for g,v in genes.items()}
    groups={k:[s for s in samples if mapper(s)==k] for k in {mapper(s) for s in samples}}
    return acc,samples,genes,groups,comparisons

def compare(dataset):
    acc,samples,genes,groups,comparisons=dataset;rows=[]
    for module,(present,score) in scores(genes,samples).items():
        for left,right in comparisons:
            a=[score[s] for s in groups[left]];b=[score[s] for s in groups[right]];mean=float(np.mean(a)-np.mean(b))
            rows.append({"accession":acc,"module":module,"genes_present":len(present),"comparison":f"{left}_vs_{right}","left_n":len(a),"right_n":len(b),"mean_difference":mean,"median_difference":float(np.median(a)-np.median(b)),"permutation_p":perm(a,b),"direction_supports_fibrotic_membrane":mean>0})
    return rows

def main():
    probe=annotation();datasets=[gse179603(),series("GSE60436",lambda s:"PDR_FVM" if s.startswith("GSM14799") and int(s[-2:])>=7 else "retina",[("PDR_FVM","retina")],probe),series("GSE41019",lambda s:"PVR" if s in {"GSM1006859","GSM1006860","GSM1006861"} else "retina",[("PVR","retina")],probe)]
    rows=sum((compare(d) for d in datasets),[]);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"bulk_module_validation.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    lines=["# External bulk validation of predefined fibrosis modules","","Positive mean differences indicate higher module scores in PVR/PDR membranes than the listed comparator.","","| Dataset | Module | Comparison | Genes | Mean difference | Permutation p | Supports membrane fibrosis |","|---|---|---|---:|---:|---:|---|"]
    lines += [f"| {r['accession']} | {r['module']} | {r['comparison']} | {r['genes_present']} | {r['mean_difference']:.3f} | {r['permutation_p']:.4f} | {r['direction_supports_fibrotic_membrane']} |" for r in rows]
    (OUT/"bulk_module_validation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":main()
