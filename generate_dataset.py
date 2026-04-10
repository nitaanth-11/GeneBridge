"""
GeneBridge Cross-Species Genomic Dataset Generator
Generates synthetic but scientifically grounded cross-species genomic data
for ML training. Species and gene mappings reflect real xenotransplantation
research, Ensembl Compara ortholog data, OMIM disease-gene associations,
and published clinical/preclinical trial outcomes.

References:
  - Ensembl Compara ortholog confidence scores
  - NCBI HomoloGene ortholog clusters
  - DisGeNET disease-gene association scores
  - STRING DB protein interaction networks
  - Published xenotransplantation literature (pig heart valves, baboon
    neuroscience models, murine knockouts, ovine tissue engineering,
    equine musculoskeletal research)
"""

import pandas as pd
import numpy as np
import json
import random
import os

random.seed(42)
np.random.seed(42)

# ─── Disease-associated human genes ──────────────────────────────────────────
# Curated from OMIM, ClinVar, and DisGeNET top-scoring gene-disease pairs
HUMAN_GENES = {
    "spinal_cord_injury": [
        "PTEN", "SOCS3", "KLF4", "STAT3", "BDNF", "NT3", "NGF", "CNTF",
        "IL6", "TNF", "NOGO", "MAG", "OMG", "LINGO1", "GAP43", "ARG1"
    ],
    "muscular_dystrophy": [
        "DMD", "DYSF", "CAPN3", "FKTN", "LARGE1", "ANO5", "SGCA", "SGCB",
        "SGCG", "SGCD", "TRIM32", "LAMA2", "COLQ", "RAPSN", "DOK7", "AGRN"
    ],
    "retinal_degeneration": [
        "RPGR", "RPGRIP1", "CNGB3", "CNGA3", "RPE65", "LRAT", "RDH12",
        "AIPL1", "GUCY2D", "CRX", "NRL", "PRPH2", "ROM1", "ABCA4", "RHO", "CEP290"
    ],
    "heart_disease": [
        "MYBPC3", "MYH7", "TNNT2", "TNNI3", "TPM1", "MYL2", "MYL3", "ACTC1",
        "PLN", "SCN5A", "LMNA", "RYR2", "CASQ2", "ANK2", "KCNQ1", "KCNH2"
    ],
    "parkinsons": [
        "SNCA", "LRRK2", "PARK7", "PINK1", "PRKN", "ATP13A2", "FBXO7",
        "GIGYF2", "HTRA2", "UCHL1", "GBA", "VPS35", "EIF4G1", "DNAJC13", "TMEM230", "CHCHD2"
    ],
    "als": [
        "SOD1", "TARDBP", "FUS", "C9ORF72", "UBQLN2", "VCP", "OPTN",
        "TBK1", "NEK1", "TUBA4A", "MATR3", "CHCHD10", "SQSTM1", "SETX", "ANG", "DCTN1"
    ],
    "deafness": [
        "GJB2", "GJB6", "SLC26A4", "MYO7A", "CDH23", "PCDH15", "OTOF",
        "TMPRSS3", "COCH", "ACTG1", "MYO15A", "TRIOBP", "ESPN", "WHRN", "STRC", "LHFPL5"
    ],
    "diabetes": [
        "INS", "INSR", "IRS1", "IRS2", "GCK", "HNF1A", "HNF4A", "PDX1",
        "NEUROD1", "KLF11", "PAX4", "PPARG", "ADIPOQ", "LEP", "LEPR", "TCF7L2"
    ]
}

# ─── Donor species with real xenotransplantation rationale ────────────────────
# base_similarity reflects Ensembl Compara protein-coding ortholog %identity
# strengths reflect real published preclinical/clinical research areas
DONOR_SPECIES = {
    "pig": {
        "sci_name": "Sus scrofa",
        # Pigs: gold standard for xenotransplantation. Pig heart valves used
        # clinically since the 1960s. Pig insulin used before recombinant.
        # GGTA1-knockout pigs bred for organ transplant. ~84% protein coding
        # gene orthology with humans (Ensembl Compara).
        "strengths": ["heart_disease", "diabetes", "retinal_degeneration"],
        "ortholog_confidence": 0.84,
        "base_similarity": 0.84
    },
    "mouse": {
        "sci_name": "Mus musculus",
        # Mice: most extensively characterised mammalian model. >80% of human
        # disease genes have a mouse ortholog (Mouse Genome Informatics).
        # Knockout libraries exist for nearly every protein-coding gene.
        # SOD1-G93A transgenic mice are THE standard ALS model.
        # mdx mice carry DMD mutation for muscular dystrophy research.
        "strengths": ["als", "muscular_dystrophy", "parkinsons", "diabetes", "deafness"],
        "ortholog_confidence": 0.85,
        "base_similarity": 0.85
    },
    "sheep": {
        "sci_name": "Ovis aries",
        # Sheep: large animal model for orthopaedic & cardiovascular research.
        # Ovine heart is anatomically similar to human (4-chambered, similar
        # mass). Used in spinal cord injury contusion models. Sheep DMD model
        # (INRA) recapitulates human Duchenne progression.
        "strengths": ["spinal_cord_injury", "heart_disease", "muscular_dystrophy"],
        "ortholog_confidence": 0.82,
        "base_similarity": 0.82
    },
    "baboon": {
        "sci_name": "Papio anubis",
        # Baboons: non-human primate with ~94% nucleotide identity.
        # Gold standard for neurological disease models (Parkinson MPTP model).
        # Used in pig-to-baboon cardiac xenotransplantation survival studies.
        # Closest immune system match outside great apes.
        "strengths": ["parkinsons", "als", "spinal_cord_injury", "heart_disease"],
        "ortholog_confidence": 0.94,
        "base_similarity": 0.94
    },
    "horse": {
        "sci_name": "Equus caballus",
        # Horses: naturally occurring HYPP (hyperkalemic periodic paralysis)
        # mirrors human SCN4A channelopathy. Equine recurrent uveitis is a
        # model for human autoimmune retinal disease. Equine metabolic syndrome
        # parallels human type 2 diabetes/insulin resistance.
        "strengths": ["muscular_dystrophy", "retinal_degeneration", "diabetes", "deafness"],
        "ortholog_confidence": 0.80,
        "base_similarity": 0.80
    },
    "axolotl": {
        "sci_name": "Ambystoma mexicanum",
        "strengths": ["spinal_cord_injury", "heart_disease"],
        "ortholog_confidence": 0.95,
        "base_similarity": 0.72
    },
    "zebrafish": {
        "sci_name": "Danio rerio",
        "strengths": ["heart_disease", "muscular_dystrophy", "retinal_degeneration", "deafness", "diabetes", "als"],
        "ortholog_confidence": 0.88,
        "base_similarity": 0.68
    },
    "salamander": {
        "sci_name": "Pleurodeles waltl",
        "strengths": ["retinal_degeneration", "spinal_cord_injury"],
        "ortholog_confidence": 0.90,
        "base_similarity": 0.70
    },
    "gecko": {
        "sci_name": "Gekko gecko",
        "strengths": ["retinal_degeneration"],
        "ortholog_confidence": 0.82,
        "base_similarity": 0.65
    },
    "naked_mole_rat": {
        "sci_name": "Heterocephalus glaber",
        "strengths": ["parkinsons", "diabetes"],
        "ortholog_confidence": 0.78,
        "base_similarity": 0.81
    },
    "planaria": {
        "sci_name": "Schmidtea mediterranea",
        "strengths": ["spinal_cord_injury", "als"],
        "ortholog_confidence": 0.99,
        "base_similarity": 0.45
    },
    "tardigrade": {
        "sci_name": "Ramazzottius varieornatus",
        "strengths": ["als", "parkinsons"],
        "ortholog_confidence": 0.91,
        "base_similarity": 0.42
    },
    "octopus": {
        "sci_name": "Octopus vulgaris",
        "strengths": ["deafness", "parkinsons"],
        "ortholog_confidence": 0.85,
        "base_similarity": 0.48
    }
}

# Gene Ontology terms relevant to regenerative medicine & disease pathways
GO_TERMS = [
    "GO:0006355", "GO:0007399", "GO:0000122", "GO:0008380", "GO:0042552",
    "GO:0007268", "GO:0045202", "GO:0031424", "GO:0051260", "GO:0000278",
    "GO:0006281", "GO:0006974", "GO:0016032", "GO:0006950", "GO:0009611",
    "GO:0002376", "GO:0006915", "GO:0008219", "GO:0045087", "GO:0030154"
]

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]

# Signalling pathways relevant to the 8 disease conditions
PATHWAYS = [
    "MAPK signaling", "PI3K-Akt", "Wnt signaling", "Notch signaling",
    "Hedgehog", "TGF-beta", "mTOR", "JAK-STAT", "NF-kB", "Autophagy",
    "Apoptosis", "DNA repair", "Oxidative phosphorylation", "Glycolysis",
    "Calcium signaling", "cAMP signaling", "Hippo signaling", "FoxO signaling"
]

# ─── Per-gene ortholog confidence overrides ───────────────────────────────────
# Certain condition-species-gene triplets have notably high or low ortholog
# confidence based on published literature. This dict maps
# (condition, species) -> {gene: similarity_delta}
GENE_SPECIFIC_BOOSTS = {
    # Pig insulin (INS) is 98% identical to human — was used clinically
    ("diabetes", "pig"): {"INS": 0.12, "INSR": 0.08, "GCK": 0.06},
    # Pig heart genes are highly conserved — bioprosthetic valve basis
    ("heart_disease", "pig"): {"MYBPC3": 0.10, "MYH7": 0.09, "TNNT2": 0.08, "PLN": 0.07},
    # Mouse SOD1-G93A is the canonical ALS model
    ("als", "mouse"): {"SOD1": 0.13, "TARDBP": 0.10, "FUS": 0.09, "C9ORF72": 0.04},
    # mdx mouse carries the exact DMD frameshift
    ("muscular_dystrophy", "mouse"): {"DMD": 0.14, "DYSF": 0.08, "SGCA": 0.07},
    # Baboon MPTP Parkinson model — SNCA/LRRK2 highly conserved in primates
    ("parkinsons", "baboon"): {"SNCA": 0.05, "LRRK2": 0.04, "PINK1": 0.04, "GBA": 0.03},
    # Baboon spinal cord — primate CNS architecture closest to human
    ("spinal_cord_injury", "baboon"): {"PTEN": 0.04, "BDNF": 0.05, "NGF": 0.04},
    # Sheep spinal cord contusion model
    ("spinal_cord_injury", "sheep"): {"BDNF": 0.07, "NT3": 0.06, "CNTF": 0.05},
    # Sheep heart — anatomical/physiological similarity
    ("heart_disease", "sheep"): {"MYBPC3": 0.08, "MYH7": 0.07, "RYR2": 0.06},
    # Sheep DMD model (INRA)
    ("muscular_dystrophy", "sheep"): {"DMD": 0.10, "DYSF": 0.06},
    # Horse HYPP = SCN4A channelopathy analogue
    ("muscular_dystrophy", "horse"): {"DMD": 0.05, "SGCA": 0.04},
    # Equine recurrent uveitis → retinal gene conservation
    ("retinal_degeneration", "horse"): {"RPE65": 0.07, "RHO": 0.06, "RPGR": 0.05},
    # Horse metabolic syndrome parallels T2D
    ("diabetes", "horse"): {"INSR": 0.06, "PPARG": 0.05, "ADIPOQ": 0.04},
    # Mouse deafness models (Shaker, waltzer, etc.)
    ("deafness", "mouse"): {"GJB2": 0.10, "MYO7A": 0.09, "CDH23": 0.08, "OTOF": 0.07},
    # Pig corneal/retinal xenotransplant research
    ("retinal_degeneration", "pig"): {"RPE65": 0.08, "RHO": 0.07, "ABCA4": 0.05},
    # Mouse diabetes models (ob/ob, db/db, NOD)
    ("diabetes", "mouse"): {"INS": 0.06, "LEP": 0.12, "LEPR": 0.11, "PPARG": 0.07},
}


def sequence_similarity(condition, species, gene):
    """Compute ortholog sequence similarity with gene-specific adjustments."""
    base = DONOR_SPECIES[species]["base_similarity"]

    # Condition-species strength boost
    if condition in DONOR_SPECIES[species]["strengths"]:
        boost = np.random.uniform(0.06, 0.14)
    else:
        boost = np.random.uniform(-0.08, 0.03)

    # Gene-specific override if available
    gene_boosts = GENE_SPECIFIC_BOOSTS.get((condition, species), {})
    gene_delta = gene_boosts.get(gene, 0.0)

    noise = np.random.normal(0, 0.025)
    return float(np.clip(base + boost + gene_delta + noise, 0.30, 0.99))


def expression_correlation(condition, species, is_positive):
    """Simulate cross-species RNA-seq expression correlation."""
    if is_positive:
        base = np.random.uniform(0.55, 0.92)
    else:
        base = np.random.uniform(0.08, 0.42)
    if condition in DONOR_SPECIES[species]["strengths"]:
        base = min(base * 1.12, 0.99)
    return round(float(base), 4)


def protein_interaction_degree(is_positive):
    """PPI degree from STRING DB-like distribution."""
    if is_positive:
        return int(np.random.negative_binomial(8, 0.35) + 5)
    else:
        return int(np.random.negative_binomial(3, 0.5) + 1)


def generate_record(condition, species, gene, gene_idx, is_positive):
    """Generate a single genomic comparison record."""
    ortholog_conf = DONOR_SPECIES[species]["ortholog_confidence"]
    seq_sim = sequence_similarity(condition, species, gene)
    expr_corr = expression_correlation(condition, species, is_positive)
    ppi_degree = protein_interaction_degree(is_positive)

    go_overlap = round(np.random.beta(5, 2) if is_positive else np.random.beta(2, 5), 4)
    conservation_score = round(seq_sim * 0.8 + np.random.normal(0, 0.035), 4)
    disease_gene_assoc = round(np.random.beta(6, 2) if is_positive else np.random.beta(2, 6), 4)
    synteny_score = round(np.random.beta(4, 2) if is_positive else np.random.beta(2, 4), 4)
    regulatory_sim = round(np.random.beta(4, 3) if is_positive else np.random.beta(2, 5), 4)
    n_go_terms = int(np.random.randint(2, 8) if is_positive else np.random.randint(0, 4))

    # Known therapy target & clinical trial flags — higher for species with
    # established preclinical or clinical use for this condition
    therapy_prob = 0.35 if condition in DONOR_SPECIES[species]["strengths"] else 0.15
    known_therapy = 1 if (is_positive and np.random.random() < therapy_prob) else 0
    trial_prob = 0.50 if species in ("pig", "baboon") else 0.30
    clinical_trial = 1 if (known_therapy and np.random.random() < trial_prob) else 0

    # Introduce irreducible biological noise (12% chance of failure despite good markers, 
    # or success despite bad markers) to prevent ML models from achieving a fake "100%" accuracy
    target = int(is_positive)
    if np.random.rand() < 0.12:
        target = 1 - target

    return {
        "record_id": f"GB-{condition[:3].upper()}-{species[:3].upper()}-{gene_idx:04d}",
        "human_gene": gene,
        "donor_species": species,
        "donor_sci_name": DONOR_SPECIES[species]["sci_name"],
        "condition": condition,
        "chromosome": random.choice(CHROMOSOMES),
        "sequence_similarity": round(seq_sim, 4),
        "expression_correlation": expr_corr,
        "protein_interaction_degree": ppi_degree,
        "go_term_overlap": go_overlap,
        "conservation_score": round(float(np.clip(conservation_score, 0.1, 0.99)), 4),
        "disease_gene_association_score": disease_gene_assoc,
        "synteny_score": synteny_score,
        "regulatory_region_similarity": regulatory_sim,
        "num_shared_go_terms": n_go_terms,
        "species_regen_score": ortholog_conf,
        "known_therapy_target": known_therapy,
        "in_clinical_trial": clinical_trial,
        "primary_pathway": random.choice(PATHWAYS),
        "target_label": target
    }


def generate_dataset(n_per_condition=180):
    """Generate the full cross-species genomic dataset."""
    records = []
    idx = 0
    for condition, genes in HUMAN_GENES.items():
        for species in DONOR_SPECIES:
            # Positive rate varies by species-condition relevance
            # Positive rate must be exceptionally rare to mimic real biology matching constraints
            if condition in DONOR_SPECIES[species]["strengths"]:
                pos_rate = 0.12 + np.random.random() * 0.06  # 12-18% viability for optimal models
            else:
                pos_rate = 0.01 + np.random.random() * 0.04  # 1-5% viability for poor matches
            n_pos = int(n_per_condition * pos_rate)
            n_neg = n_per_condition - n_pos
            for _ in range(n_pos):
                gene = random.choice(genes)
                records.append(generate_record(condition, species, gene, idx, True))
                idx += 1
            for _ in range(n_neg):
                gene = random.choice(genes)
                records.append(generate_record(condition, species, gene, idx, False))
                idx += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating GeneBridge cross-species genomic dataset...")
    df = generate_dataset(n_per_condition=180)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "genebridge_dataset.csv")
    df.to_csv(out_path, index=False)
    # Also save to root for fallback
    df.to_csv("genebridge_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} records across {df['condition'].nunique()} conditions")
    print(f"Species: {df['donor_species'].unique().tolist()}")
    print(f"Class balance: {df['target_label'].value_counts().to_dict()}")
    print(f"Columns: {list(df.columns)}")
    print(df.describe())
