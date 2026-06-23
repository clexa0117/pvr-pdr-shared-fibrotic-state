import csv
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"results/stop_go_prescreen"
def read(name):
    with (BASE/name).open(encoding="utf-8",newline="") as h:return list(csv.DictReader(h))
def med(rows,key,disease):return statistics.median(float(r[key]) for r in rows if r["disease"]==disease)
def stable(rows,key,expected):
    return sum((med([x for x in rows if x is not r],key,"PVR")>med([x for x in rows if x is not r],key,"PDR"))==expected for r in rows)
def main():
    samples=read("sample_qc_and_modules.csv");bulk=read("bulk_module_validation.csv");shared=read("shared_module_screen.csv")
    metrics=[("rpe_origin_positive_pct_fibrotic","RPE-like origin signal",True),("pericyte_origin_positive_pct_fibrotic","Pericyte/vascular-mural origin signal",False),("angiogenesis_positive_pct_fibrotic","Angiogenesis signal",False)]
    origins=[(label,med(samples,key,"PVR"),med(samples,key,"PDR"),stable(samples,key,expected),len(samples)) for key,label,expected in metrics]
    core=[m for m in ("fibrosis_ecm","contractile","tgfb_mechanosensing") if all(float(r["median_positive_pct_in_fibrotic_cells"])>=50 for r in shared if r["module"]==m)]
    pvr=any(r["module"]=="fibrosis_ecm" and r["accession"] in {"GSE179603","GSE41019"} and r["direction_supports_fibrotic_membrane"]=="True" for r in bulk)
    pdr=any(r["module"]=="fibrosis_ecm" and r["accession"]=="GSE60436" and r["direction_supports_fibrotic_membrane"]=="True" for r in bulk)
    candidates=all(int(r["fibrotic_candidate_cells"])>=50 for r in samples);origin=origins[0][1]>origins[0][2] and origins[1][1]<origins[1][2];go=candidates and len(core)>=2 and pvr and pdr and origin
    lines=["# Final stop/go decision for the PVR-PDR fibrosis project","",f"## Decision: {'GO - continue to full preprocessing and formal analysis' if go else 'STOP - revise or replace the project'}","","## Evidence against the predefined stop criteria","",f"- All 15 human single-cell samples contain at least 50 candidate fibrotic cells: {candidates}",f"- Shared core modules passing the broad cross-disease screen: {', '.join(core)}",f"- Fibrosis ECM program supported by independent PVR bulk data: {pvr}",f"- Fibrosis ECM program supported by independent PDR bulk data: {pdr}",f"- Divergent-origin signal supports the proposed story: {origin}","","## Disease-origin signal in candidate fibrotic cells","","| Signal | PVR median positive percent | PDR median positive percent | Expected direction stable after leaving out each sample |","|---|---:|---:|---:|"]
    lines += [f"| {label} | {a:.1f}% | {b:.1f}% | {s}/{n} |" for label,a,b,s,n in origins]
    (BASE/"FINAL_STOP_GO_DECISION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":main()
