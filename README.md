# GeneBridge — Cross-Species Genetics ML Platform

Mapping human disease genes to cross-species orthologs using Machine Learning.  
RF + Gradient Boosting + Calibrated SVM Ensemble · 11,520 training records · 8 conditions · 8 donor species

---

## Deployment on Render

This project is fully structured to be deployed on [Render](https://render.com/) with a single click or Blueprint configuration.

### Option 1: Blueprints Deployment (Recommended)
1. Commit and push this project to your GitHub repository.
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** and select **Blueprint**.
4. Connect your repository. Render will read the `render.yaml` configuration file automatically and provision your Flask Web Service.

### Option 2: Manual Deployment
1. Create a new **Web Service** on Render.
2. Connect your Git repository.
3. Configure the following settings:
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt && python -c "import genebridge_backend"`
   - **Start Command**: `gunicorn genebridge_backend:app`
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.10.12` (or your preferred version)

---

## 🔬 What is GeneBridge?
Nature has already solved many diseases we struggle with:
- **Axolotls** regrow their entire spinal cord after injury
- **Zebrafish** regenerate heart muscle after ventricular resection
- **Tardigrades** protect their neurons under extreme radiation
- **Naked mole rats** show near-zero cancer and neurodegeneration rates

GeneBridge uses a 3-model ML ensemble to rank how viable each of these cross-species gene mappings is as a human therapy target — given a patient's condition, age, and disease severity.

## 🖥️ Demo
- **Before therapy**: Affected body regions pulse red on the 3D model.
- **After therapy**: Regions switch to green with confirmation marks.
- **Species cards**: 8 donor species ranked by viability score with animated score bars.
- **Gene chips**: Click any gene to view detailed statistics in a modal.

## 🏗️ Architecture
```
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
```

## 📁 File Structure
```
genebridge/
├── genebridge_backend.py          # Flask API + ML training + model serving
├── generate_dataset.py            # Synthetic dataset generator
├── genebridge_frontend.html       # Complete single-file frontend (HTML+CSS+JS)
├── data/
│   └── genebridge_dataset.csv     # 11,520 training records (auto-generated)
├── models/
│   └── genebridge_models.pkl      # Saved model artifacts (auto-generated)
├── requirements.txt               # Python dependencies
├── render.yaml                    # Render Blueprint config
└── README.md
```

## ⚙️ Setup & Installation (Local)
### Prerequisites
- Python 3.9+
- Any modern browser (Chrome, Firefox, Edge, Safari)

1. Clone the repo
   ```bash
   git clone https://github.com/nitaanth-11/GeneBridge.git
   cd GeneBridge
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Project Locally
1. Start the backend:
   ```bash
   python genebridge_backend.py
   ```
   *Note: Flask automatically serves the interactive 3D Body Map frontend (`genebridge_frontend.html`) at the root URL (`/`).*

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 🤖 ML Pipeline
### Model 1 — Random Forest (45% weight)
Builds 300 independent decision trees on random data/feature subsets, aggregates by majority vote.
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
```
### Model 2 — Gradient Boosting (35% weight)
Builds 200 trees sequentially — each tree corrects the errors of the previous one.
```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
```
### Model 3 — Calibrated SVM (20% weight)
Finds an optimal hyperplane in 15-dimensional feature space. Isotonic regression calibrates raw scores to probabilities.
```python
CalibratedClassifierCV(
    SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced'),
    method='isotonic',
    cv=3
)
```
### Ensemble Formula
```
viability = (0.45 × P_rf) + (0.35 × P_gb) + (0.20 × P_svm)

# Clinical adjustment:
viability = viability × severity_multiplier − age_penalty
# severity_multiplier: mild=1.05, moderate=1.00, severe=0.88
# age_penalty:         max(0, (age − 25) × 0.003)
```

## 📊 Dataset
- **Total records**: 11,520
- **Conditions**: 8
- **Donor species**: 8
- **Records per condition × species**: 180
- **Class balance**: 40% viable / 60% non-viable
- **Numeric features**: 12
- **Categorical features**: 3 (label-encoded)
- **Total features**: 15

### Features
- `sequence_similarity`: BLAST-style alignment score (0–1)
- `expression_correlation`: RNA-seq similarity in disease tissue
- `protein_interaction_degree`: Number of protein-protein interactions
- `go_term_overlap`: Fraction of shared Gene Ontology terms
- `conservation_score`: Evolutionary conservation across taxa
- `disease_gene_association_score`: Strength of disease-gene link
- `synteny_score`: Chromosomal gene-order conservation
- `regulatory_region_similarity`: Promoter/enhancer region similarity
- `num_shared_go_terms`: Count of shared GO annotations
- `species_regen_score`: Donor species regenerative capacity
- `known_therapy_target`: Binary: known therapy target?
- `in_clinical_trial`: Binary: active clinical trial?

## 🌐 API Reference
- **Base URL**: `/api`
- **Endpoints**:
  - `GET /health` - API status and loaded models
  - `GET /metrics` - All model performance metrics
  - `GET /conditions` - Supported conditions with metadata
  - `GET /species` - Donor species with metadata
  - `POST /predict` - Main prediction — ranked species results
  - `POST /gene-detail` - Stats for a specific gene/condition/species

## 🔭 Real-World Data Sources (for production upgrade)
- NCBI HomoloGene: Homology mappings
- Ensembl Compara: Gene trees and synteny blocks
- DisGeNET: Disease-gene associations
- NCBI GEO: Gene expression datasets
- STRING DB: Protein-protein interaction networks
- UniProt: Functional annotations

## ⚠️ Disclaimer
GeneBridge is a research prototype. All predictions are based on synthetic training data and must not be used for clinical decision-making. Always consult qualified medical and genomics professionals.

## 📄 License
MIT License — free to use, modify, and distribute with attribution.
