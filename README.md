GeneBridge — Cross-Species Genetics ML Platform

Mapping human disease genes to cross-species orthologs using Machine Learning
RF + Gradient Boosting + Calibrated SVM Ensemble · 11,520 training records · 8 conditions · 8 donor species


🔬 What is GeneBridge?
Nature has already solved many diseases we struggle with:

Axolotls regrow their entire spinal cord after injury
Zebrafish regenerate heart muscle after ventricular resection
Tardigrades protect their neurons under extreme radiation
Naked mole rats show near-zero cancer and neurodegeneration rates

GeneBridge uses a 3-model ML ensemble to rank how viable each of these cross-species gene mappings is as a human therapy target — given a patient's condition, age, and disease severity.

🖥️ Demo
ViewDescriptionBefore therapyAffected body regions pulse red on the 3D modelAfter therapyRegions switch to green with confirmation marksSpecies cards8 donor species ranked by viability score with animated score barsGene chipsClick any gene to view detailed statistics in a modal

🏗️ Architecture
Browser (index.html)
    │
    ▼  HTTP POST /api/predict
Flask REST API  (backend.py · port 5000)
    │
    ├──▶  Random Forest      (weight: 45%)
    ├──▶  Gradient Boosting  (weight: 35%)
    └──▶  Calibrated SVM     (weight: 20%)
               │
               ▼
    genebridge_dataset.csv   (11,520 records)
    genebridge_models.pkl    (saved sklearn artifacts)

📁 File Structure
genebridge/
├── backend.py                  # Flask API + ML training + model serving
├── generate_dataset.py         # Synthetic dataset generator
├── frontend/
│   └── index.html              # Complete single-file frontend (HTML+CSS+JS)
├── data/
│   └── genebridge_dataset.csv  # 11,520 training records (auto-generated)
├── models/
│   └── genebridge_models.pkl   # Saved model artifacts (auto-generated)
└── README.md

⚙️ Setup & Installation
Prerequisites

Python 3.9+
Any modern browser (Chrome, Firefox, Edge, Safari)

1. Clone the repo
bashgit clone https://github.com/YOUR_USERNAME/genebridge.git
cd genebridge
2. Install dependencies
bashpip install flask scikit-learn pandas numpy joblib

No XGBoost needed — the project uses scikit-learn's GradientBoostingClassifier.


🚀 Running the Project
Step 1 — Generate the dataset
bashpython generate_dataset.py
# Creates: data/genebridge_dataset.csv (11,520 records)
Step 2 — Train models & start the backend
bashpython backend.py
# First run: trains all 3 models (~60 sec), saves to models/
# Subsequent runs: loads saved models instantly
# API live at http://localhost:5000
Step 3 — Open the frontend
bash# Just open index.html in your browser — no build step needed
open frontend/index.html          # macOS
start frontend/index.html         # Windows
xdg-open frontend/index.html      # Linux
Step 4 — Use the app

Select a condition from the dropdown
Set age, severity, and biological sex
Click Analyse Genetic Candidates
Explore ranked species cards and top target genes
Toggle the 3D body between Before and After therapy
Click any gene chip for detailed stats


Demo mode: If the backend isn't running, the frontend auto-switches to simulated results — no setup needed to explore the UI.


🤖 ML Pipeline
Model 1 — Random Forest (45% weight)
Builds 300 independent decision trees on random data/feature subsets, aggregates by majority vote.
pythonRandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
Model 2 — Gradient Boosting (35% weight)
Builds 200 trees sequentially — each tree corrects the errors of the previous one.
pythonGradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
Model 3 — Calibrated SVM (20% weight)
Finds an optimal hyperplane in 15-dimensional feature space. Isotonic regression calibrates raw scores to probabilities.
pythonCalibratedClassifierCV(
    SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced'),
    method='isotonic',
    cv=3
)
Ensemble Formula
viability = (0.45 × P_rf) + (0.35 × P_gb) + (0.20 × P_svm)

# Clinical adjustment:
viability = viability × severity_multiplier − age_penalty
# severity_multiplier: mild=1.05, moderate=1.00, severe=0.88
# age_penalty:         max(0, (age − 25) × 0.003)
Performance (on synthetic test set)
ModelAccuracyROC-AUCRandom Forest99.98%1.0000Gradient Boosting99.98%1.0000Calibrated SVM99.98%1.0000Ensemble99.98%1.0000

AUC = 1.0 is expected on synthetic data with clear class separation. Real-world genomic data would yield AUC ≈ 0.78–0.92.


📊 Dataset
PropertyValueTotal records11,520Conditions8Donor species8Records per condition×species180Class balance40% viable / 60% non-viableNumeric features12Categorical features3 (label-encoded)Total features15
Features
FeatureDescriptionInspired bysequence_similarityBLAST-style alignment score (0–1)NCBI BLASTexpression_correlationRNA-seq similarity in disease tissueNCBI GEOprotein_interaction_degreeNumber of protein-protein interactionsSTRING DBgo_term_overlapFraction of shared Gene Ontology termsUniProt / GOconservation_scoreEvolutionary conservation across taxaEnsembl PhyloPdisease_gene_association_scoreStrength of disease-gene linkDisGeNETsynteny_scoreChromosomal gene-order conservationEnsembl Compararegulatory_region_similarityPromoter/enhancer region similarityENCODE / JASPARnum_shared_go_termsCount of shared GO annotationsGO Consortiumspecies_regen_scoreDonor species regenerative capacityLiteratureknown_therapy_targetBinary: known therapy target?DrugBankin_clinical_trialBinary: active clinical trial?ClinicalTrials.gov

🌐 API Reference
Base URL: http://localhost:5000/api
MethodEndpointDescriptionGET/healthAPI status and loaded modelsGET/metricsAll model performance metricsGET/conditionsSupported conditions with metadataGET/speciesDonor species with metadataPOST/predictMain prediction — ranked species resultsPOST/gene-detailStats for a specific gene/condition/species
POST /api/predict
Request:
json{
  "condition": "spinal_cord_injury",
  "age": 42,
  "severity": "moderate",
  "gender": "male"
}
Response:
json{
  "summary": {
    "viable_species_count": 4,
    "best_species": "axolotl",
    "best_viability": 0.8742,
    "best_genes": ["PTEN", "SOCS3", "BDNF"]
  },
  "results": [
    {
      "species": "axolotl",
      "viability_score": 0.8742,
      "top_genes": ["PTEN", "SOCS3", "BDNF", "KLF4", "STAT3"],
      "primary_pathway": "PI3K-Akt",
      "is_recommended": true
    }
  ]
}

🦎 Supported Conditions & Species
Conditions
ConditionKey GenesBest Donor SpeciesSpinal Cord InjuryPTEN, SOCS3, KLF4, BDNFAxolotl, Salamander, PlanariaMuscular DystrophyDMD, DYSF, CAPN3, SGCAZebrafish, SalamanderRetinal DegenerationRPGR, RPE65, CNGA3, CEP290Gecko, Salamander, ZebrafishHeart DiseaseMYBPC3, MYH7, TNNT2, PLNZebrafish, AxolotlParkinson's DiseaseSNCA, LRRK2, PINK1, PRKNNaked Mole Rat, TardigradeALS / MNDSOD1, TARDBP, FUS, C9ORF72Tardigrade, PlanariaSensorineural DeafnessGJB2, MYO7A, OTOF, CDH23Octopus, SalamanderDiabetes (T1/T2)INS, INSR, GCK, HNF1ANaked Mole Rat, Zebrafish
Donor Species
SpeciesAbilityRegen Score🦎 AxolotlFull spinal cord & limb regeneration0.95🐟 ZebrafishHeart muscle & retinal regeneration0.88🦎 GeckoTail & photoreceptor regeneration0.82🐀 Naked Mole RatNeuroprotection & cancer resistance0.78🪱 PlanariaFull body regeneration from any cell0.99🔬 TardigradeExtreme stress protein (Dsup) expression0.91🐙 OctopusArm nerve regeneration0.85🦎 SalamanderRetina, limb & heart regeneration0.90

🔭 Real-World Data Sources (for production upgrade)
Replace the synthetic dataset with data from:
SourceURLDataNCBI HomoloGenencbi.nlm.nih.gov/homologeneHuman-animal gene mappingsEnsembl Comparaensembl.orgGene trees, synteny blocksDisGeNETdisgenet.orgDisease-gene associationsNCBI GEOncbi.nlm.nih.gov/geoGene expression by diseaseSTRING DBstring-db.orgProtein interaction networksUniProtuniprot.orgProtein function annotationsENCODEencodeproject.orgRegulatory region dataClinicalTrials.govclinicaltrials.govActive trial flags

🛣️ Roadmap

 Replace synthetic data with real NCBI / Ensembl / DisGeNET data
 Add SHAP explainability for per-prediction feature importance
 Integrate live NCBI BLAST API for real-time sequence scoring
 VCF file upload for personalised patient genetic profiling
 D3.js gene pathway network visualisation
 Drug interaction checker for identified gene targets
 Expand to 20+ conditions and 15+ donor species
 Bootstrap confidence intervals on viability scores
 PDF clinical report export
 Comparative genomics browser with gene alignment view


⚠️ Disclaimer
GeneBridge is a research prototype. All predictions are based on synthetic training data and must not be used for clinical decision-making. Always consult qualified medical and genomics professionals.

📄 License
MIT License — free to use, modify, and distribute with attribution.
