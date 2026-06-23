import csv
import gzip
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/validation"
CORE_PATH = ROOT / "results/patient_pseudobulk/robust_shared_state_genes.csv"
OUT = ROOT / "results/external_validation_targets"

TARGET_CLASSES = {
    "ECM_structural": {"COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL5A1", "COL8A1", "BGN", "DCN", "LUM", "VCAN"},
    "Secreted_or_extracellular_regulator": {"FN1", "POSTN", "THBS2", "CTHRC1", "IGFBP7", "CFH", "TIMP1", "MGP", "SULF1", "AEBP1", "SERPINE1"},
    "Intracellular_or_contractile": {"TAGLN", "FHL2", "MICAL2"},
}
INTERVENTION_PRIORITY = {
    "FN1": "high", "POSTN": "high", "THBS2": "high", "CTHRC1": "high",
    "IGFBP7": "medium", "SERPINE1": "risk_review", "SULF1": "medium",
    "AEBP1": "medium", "TIMP1": "risk_review", "CFH": "do_not_inhibit", "MGP": "low",
}
TRANSLATIONAL_TIER = {
    "POSTN": "Tier_1_direct_ocular",
    "CTHRC1": "Tier_2_mechanistic",
    "THBS2": "Tier_2_mechanistic",
    "FN1": "Tier_2_mechanistic",
    "AEBP1": "Tier_2_mechanistic",
    "SULF1": "Tier_3_emerging",
    "IGFBP7": "Tier_3_emerging",
    "SERPINE1": "Risk_review_context_dependent",
    "TIMP1": "Risk_review_context_dependent",
    "CFH": "Do_not_inhibit_retinal_protective",
}
MECHANISM_RATIONALE = {
    "POSTN": "Direct human PVR/PDR membrane evidence and experimental ocular blockade evidence.",
    "CTHRC1": "Marks pathogenic myofibroblasts and promotes TGF-beta-linked fibrosis; ocular intervention evidence is not yet direct.",
    "THBS2": "Extracellular activator of TLR4-FAK/TGF-beta fibrosis; retinal fibrosis association but no direct PVR/PDR blockade study.",
    "FN1": "Central injury-associated ECM and potentially targetable through disease-associated fibronectin domains; broad normal-matrix expression is a risk.",
    "AEBP1": "Linked to pericyte-myofibroblast transformation in PDR and promotes fibrosis in other organs.",
    "SULF1": "Emerging extracellular regulator that can promote TGF-beta/SMAD fibrosis; ocular mechanism is unvalidated.",
    "IGFBP7": "Fibrosis-associated secreted factor with emerging intervention evidence outside the eye.",
    "SERPINE1": "Present in PVR/PDR membranes and targetable, but PAI-1 can be pro- or anti-fibrotic depending on context.",
    "TIMP1": "ECM-remodeling marker with context-dependent effects; inhibition could increase matrix degradation or injury unpredictably.",
    "CFH": "Complement inhibitor with retinal protective functions; expression support must not be interpreted as rationale for inhibition.",
}
LITERATURE_URLS = {
    "POSTN": "https://pubmed.ncbi.nlm.nih.gov/24022401/; https://pubmed.ncbi.nlm.nih.gov/28913545/",
    "CTHRC1": "https://pubmed.ncbi.nlm.nih.gov/30639416/; https://pubmed.ncbi.nlm.nih.gov/38834096/",
    "THBS2": "https://pubmed.ncbi.nlm.nih.gov/38379585/; https://pubmed.ncbi.nlm.nih.gov/35039609/",
    "FN1": "https://pubmed.ncbi.nlm.nih.gov/27484779/; https://pubmed.ncbi.nlm.nih.gov/28827403/",
    "AEBP1": "https://pubmed.ncbi.nlm.nih.gov/37917183/; https://pubmed.ncbi.nlm.nih.gov/36738398/",
    "SULF1": "https://pubmed.ncbi.nlm.nih.gov/39354547/",
    "IGFBP7": "https://pubmed.ncbi.nlm.nih.gov/40381971/",
    "SERPINE1": "https://pubmed.ncbi.nlm.nih.gov/14562162/; https://pubmed.ncbi.nlm.nih.gov/40894033/; https://pubmed.ncbi.nlm.nih.gov/33432417/",
    "TIMP1": "https://pubmed.ncbi.nlm.nih.gov/24526442/; https://pubmed.ncbi.nlm.nih.gov/11004090/",
    "CFH": "https://pubmed.ncbi.nlm.nih.gov/35038170/; https://pubmed.ncbi.nlm.nih.gov/25447048/",
}


def exact_permutation_p(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    observed = abs(left.mean() - right.mean())
    combined = np.concatenate([left, right])
    n_left = len(left)
    exceed = 0
    total = 0
    for selected in itertools.combinations(range(len(combined)), n_left):
        mask = np.zeros(len(combined), bool)
        mask[list(selected)] = True
        difference = abs(combined[mask].mean() - combined[~mask].mean())
        exceed += difference >= observed - 1e-12
        total += 1
    return (exceed + 1) / (total + 1)


def hedges_g(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    pooled = np.sqrt(
        ((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1))
        / (len(left) + len(right) - 2)
    )
    if pooled == 0:
        return 0.0
    d = (left.mean() - right.mean()) / pooled
    correction = 1 - 3 / (4 * (len(left) + len(right)) - 9)
    return float(d * correction)


def annotation():
    mapping = {}
    with gzip.open(DATA / "GPL6884/GPL6884.annot.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ID\t"):
                header = line.rstrip().split("\t")
                break
        for row in csv.DictReader(handle, fieldnames=header, delimiter="\t"):
            gene = (row.get("Gene symbol") or "").split("///")[0].strip()
            if gene:
                mapping[row["ID"]] = gene
    return mapping


def load_gse179603(target):
    genes = {}
    with gzip.open(DATA / "GSE179603/GSE179603_data.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        samples = [value.strip('"') for value in header[5:32]]
        for row in reader:
            gene = row[0].strip('"')
            if gene not in target:
                continue
            values = [value.strip('"').replace(",", ".") for value in row[5:32]]
            if all(value not in {"", "NA"} for value in values):
                genes[gene] = np.asarray(list(map(float, values)))
    groups = {
        "PVR": [index for index, sample in enumerate(samples) if sample.startswith("PVR_")],
        "gliosis": [index for index, sample in enumerate(samples) if sample.startswith("Gliose_")],
        "ILM": [index for index, sample in enumerate(samples) if sample.startswith("ILM_")],
    }
    return "GSE179603", samples, genes, groups, [("PVR", "gliosis"), ("PVR", "ILM")]


def load_series(accession, target, probe_mapping):
    probes = defaultdict(list)
    samples = []
    with gzip.open(DATA / accession / f"{accession}_series_matrix.txt.gz", "rt", encoding="utf-8") as handle:
        in_table = False
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table:
                continue
            parts = [value.strip('"') for value in line.rstrip().split("\t")]
            if parts[0] == "ID_REF":
                samples = parts[1:]
                continue
            gene = probe_mapping.get(parts[0])
            if gene in target:
                probes[gene].append(np.asarray(list(map(float, parts[1:]))))
    genes = {gene: np.vstack(values).mean(axis=0) for gene, values in probes.items()}
    if accession == "GSE60436":
        groups = {"PDR_FVM": list(range(3, 9)), "retina": list(range(0, 3))}
        comparisons = [("PDR_FVM", "retina")]
    else:
        groups = {"PVR": list(range(0, 3)), "retina": list(range(3, 6))}
        comparisons = [("PVR", "retina")]
    return accession, samples, genes, groups, comparisons


def gene_validation(dataset, target):
    accession, _, genes, groups, comparisons = dataset
    rows = []
    for left_name, right_name in comparisons:
        comparison_rows = []
        for gene in sorted(target):
            if gene not in genes:
                comparison_rows.append({
                    "accession": accession, "comparison": f"{left_name}_vs_{right_name}", "gene": gene,
                    "present": False, "left_n": len(groups[left_name]), "right_n": len(groups[right_name]),
                    "mean_difference": np.nan, "median_difference": np.nan, "hedges_g": np.nan,
                    "permutation_p": np.nan, "direction_supports_membrane": False,
                })
                continue
            left = genes[gene][groups[left_name]]
            right = genes[gene][groups[right_name]]
            comparison_rows.append({
                "accession": accession, "comparison": f"{left_name}_vs_{right_name}", "gene": gene,
                "present": True, "left_n": len(left), "right_n": len(right),
                "mean_difference": float(left.mean() - right.mean()),
                "median_difference": float(np.median(left) - np.median(right)),
                "hedges_g": hedges_g(left, right),
                "permutation_p": exact_permutation_p(left, right),
                "direction_supports_membrane": bool(left.mean() > right.mean()),
            })
        valid = [row["permutation_p"] for row in comparison_rows if row["present"]]
        adjusted = multipletests(valid, method="fdr_bh")[1] if valid else []
        index = 0
        for row in comparison_rows:
            row["permutation_fdr"] = float(adjusted[index]) if row["present"] else np.nan
            index += row["present"]
        rows.extend(comparison_rows)
    return rows


def module_score(dataset, target):
    accession, _, genes, groups, comparisons = dataset
    present = sorted(target & set(genes))
    matrix = np.vstack([genes[gene] for gene in present])
    sd = matrix.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    z = (matrix - matrix.mean(axis=1, keepdims=True)) / sd
    score = z.mean(axis=0)
    rows = []
    for left_name, right_name in comparisons:
        left = score[groups[left_name]]
        right = score[groups[right_name]]
        rows.append({
            "accession": accession,
            "comparison": f"{left_name}_vs_{right_name}",
            "genes_present": len(present),
            "genes_total": len(target),
            "left_n": len(left),
            "right_n": len(right),
            "mean_difference": float(left.mean() - right.mean()),
            "hedges_g": hedges_g(left, right),
            "permutation_p": exact_permutation_p(left, right),
            "direction_supports_membrane": bool(left.mean() > right.mean()),
        })
    return rows


def target_class(gene):
    for name, genes in TARGET_CLASSES.items():
        if gene in genes:
            return name
    return "Unclassified"


def build_evidence_matrix(gene_results, core):
    table = pd.DataFrame(gene_results)
    rows = []
    for gene in core:
        subset = table[table["gene"] == gene]
        evidence = {(row.accession, row.comparison): row for row in subset.itertuples()}
        def support(accession, comparison):
            row = evidence.get((accession, comparison))
            return bool(row and row.present and row.direction_supports_membrane)
        def effect(accession, comparison):
            row = evidence.get((accession, comparison))
            return float(row.hedges_g) if row and row.present else np.nan
        pvr_179_g = support("GSE179603", "PVR_vs_gliosis")
        pvr_179_i = support("GSE179603", "PVR_vs_ILM")
        pvr_410 = support("GSE41019", "PVR_vs_retina")
        pdr_604 = support("GSE60436", "PDR_FVM_vs_retina")
        pvr_supports = sum([pvr_179_g, pvr_179_i, pvr_410])
        cross_disease = pdr_604 and pvr_supports >= 2
        classification = target_class(gene)
        priority = INTERVENTION_PRIORITY.get(gene, "not_prioritized")
        rows.append({
            "gene": gene,
            "target_class": classification,
            "intervention_priority": priority,
            "GSE179603_PVR_vs_gliosis_support": pvr_179_g,
            "GSE179603_PVR_vs_ILM_support": pvr_179_i,
            "GSE41019_PVR_vs_retina_support": pvr_410,
            "GSE60436_PDR_FVM_vs_retina_support": pdr_604,
            "PVR_support_count": pvr_supports,
            "cross_disease_external_direction_support": cross_disease,
            "effect_GSE179603_PVR_vs_gliosis": effect("GSE179603", "PVR_vs_gliosis"),
            "effect_GSE179603_PVR_vs_ILM": effect("GSE179603", "PVR_vs_ILM"),
            "effect_GSE41019_PVR_vs_retina": effect("GSE41019", "PVR_vs_retina"),
            "effect_GSE60436_PDR_FVM_vs_retina": effect("GSE60436", "PDR_FVM_vs_retina"),
            "candidate_target": bool(
                cross_disease
                and classification == "Secreted_or_extracellular_regulator"
                and priority in {"high", "medium", "risk_review", "do_not_inhibit"}
            ),
            "translational_tier": TRANSLATIONAL_TIER.get(gene, "Not_prioritized"),
            "mechanism_rationale": MECHANISM_RATIONALE.get(gene, ""),
            "literature_urls": LITERATURE_URLS.get(gene, ""),
        })
    return pd.DataFrame(rows)


def plot_effects(evidence):
    columns = [
        "effect_GSE179603_PVR_vs_gliosis",
        "effect_GSE179603_PVR_vs_ILM",
        "effect_GSE41019_PVR_vs_retina",
        "effect_GSE60436_PDR_FVM_vs_retina",
    ]
    values = evidence.set_index("gene")[columns]
    fig, ax = plt.subplots(figsize=(10, max(7, len(values) * 0.35)))
    sns.heatmap(values, cmap="vlag", center=0, annot=True, fmt=".1f", ax=ax)
    ax.set_title("External validation effect sizes (Hedges g)")
    fig.tight_layout()
    fig.savefig(OUT / "external_gene_effect_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(CORE_PATH)["gene"].tolist()
    target = set(core)
    probe = annotation()
    datasets = [
        load_gse179603(target),
        load_series("GSE60436", target, probe),
        load_series("GSE41019", target, probe),
    ]
    gene_rows = []
    module_rows = []
    for dataset in datasets:
        gene_rows.extend(gene_validation(dataset, target))
        module_rows.extend(module_score(dataset, target))
    pd.DataFrame(gene_rows).to_csv(OUT / "external_gene_validation.csv", index=False)
    pd.DataFrame(module_rows).to_csv(OUT / "external_core_module_validation.csv", index=False)

    evidence = build_evidence_matrix(gene_rows, core)
    evidence.to_csv(OUT / "external_evidence_matrix.csv", index=False)
    candidates = evidence[evidence["candidate_target"]].sort_values(
        ["translational_tier", "PVR_support_count"], ascending=[True, False]
    )
    candidates.to_csv(OUT / "candidate_non_vegf_targets.csv", index=False)
    prioritized = candidates[candidates["translational_tier"].isin({
        "Tier_1_direct_ocular", "Tier_2_mechanistic", "Tier_3_emerging",
    })].copy()
    prioritized.to_csv(OUT / "prioritized_non_vegf_targets.csv", index=False)
    candidates[~candidates.index.isin(prioritized.index)].to_csv(OUT / "target_risk_review.csv", index=False)
    plot_effects(evidence)

    module = pd.DataFrame(module_rows)
    lines = [
        "# ", "",
        f"- : {len(core)}",
        f"- : {int(evidence['cross_disease_external_direction_support'].sum())}",
        f"- : {len(candidates)}",
        f"- : {len(prioritized)}",
        "",
        "## ", "",
        module.to_markdown(index=False),
        "",
        "##  VEGF ", "",
        prioritized[[
            "gene", "translational_tier", "target_class", "PVR_support_count",
            "GSE60436_PDR_FVM_vs_retina_support",
        ]].to_markdown(index=False) if len(prioritized) else ".",
        "",
        "## ", "",
        candidates[~candidates.index.isin(prioritized.index)][[
            "gene", "translational_tier", "mechanism_rationale",
        ]].to_markdown(index=False),
        "",
        ":  PDR ,  PVR , /., .",
    ]
    (OUT / "external_validation_target_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
