# MLOps Baseline - Tabular Classification (Reproducible End-to-End Pipeline)

Reproducible, local-first ML project that mirrors a real production workflow:
**data generation -> training/evaluation -> artifact export -> audit-ready report**.

This repository was built as an **MLOps entry-level baseline** with strong emphasis on:
- deterministic runs (seeded)
- clean separation of steps (data vs training)
- traceable artifacts (models, metrics, plots, logs)
- zero large files committed to Git

---

## Why this repo matters (MLOps mindset)

In real ML systems, the model is only one piece.
What matters is the **pipeline**: repeatability, traceability, and deliverables that can be audited.

This project demonstrates:
- **Reproducible dataset creation** (same seed -> same dataset)
- **Consistent preprocessing** (imputation + scaling + one-hot encoding)
- **Cross-validation evaluation** (not only a single train/test split)
- **Artifact export** (models + metrics + plots + run logs)
- **Git hygiene** (no 800MB datasets committed)

---

## What it does

### Step 1 - Synthetic dataset generation
Creates a large tabular dataset with:
- numeric + categorical features
- realistic missing values
- binary target (`target`)

Output:
- `data/dataset_grande.csv` (ignored by Git)

### Step 2 - ML training + evaluation
Trains and evaluates two baseline models:
- Logistic Regression
- Random Forest

Evaluation includes:
- Stratified train/test split
- 5-fold Stratified CV
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion Matrix + ROC Curve plots

Outputs:
- trained models (`*.joblib`)
- plots (`*.png`)
- report (`report.json`)
- run log (`console_run.txt`)

---

## Repository structure

```text
Projeto_ML/
  gerar_csv_grande.py
  ml_pipeline.py
  requirements.txt
  README.md
  LICENSE
  .gitignore

  data/
    (generated dataset - ignored by git)

  outputs/
    logreg_confusion_matrix.png
    logreg_roc_curve.png
    logreg_model.joblib

    random_forest_confusion_matrix.png
    random_forest_roc_curve.png
    random_forest_model.joblib

    report.json
    console_run.txt
