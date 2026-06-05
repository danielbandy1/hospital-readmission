# Hospital 30-Day Readmission Prediction

![CI](https://github.com/danielbandy1/hospital-readmission/actions/workflows/ci.yml/badge.svg)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.680-blue)](#quick-results)
[![AUPRC](https://img.shields.io/badge/AUPRC-0.230-green)](#quick-results)
[![API](https://img.shields.io/badge/API-FastAPI-009688)](#deployment)

End-to-end healthcare ML system predicting whether a diabetic patient will be readmitted within 30 days of discharge: 74 engineered EHR features, LightGBM + XGBoost with Optuna tuning, SHAP explanations, calibrated risk tiers, and a FastAPI endpoint.

**Result:** LightGBM Optuna-tuned reaches **ROC-AUC 0.680** and **AUPRC 0.230** on 101,766 patient encounters. Structured EHR data caps around 0.68–0.73 AUC in the published literature; this result is honest and reproducible.

## The Problem

Hospital readmission within 30 days is one of the most studied quality metrics in US healthcare. For diabetic patients specifically, readmission rates run 15–20% — and preventable readmissions cost the US healthcare system over $17 billion annually. Identifying high-risk patients at discharge allows care teams to intervene: follow-up calls, medication reconciliation, care coordination.

This project builds a production-ready readmission scoring system that answers four questions a clinical team actually asks:

1. **Who is likely to come back within 30 days?** → LightGBM Optuna-tuned (ROC-AUC 0.680, AUPRC 0.230) and XGBoost (ROC-AUC 0.680, AUPRC 0.226)
2. **Why?** → Per-patient SHAP explanations surfaced in the API response
3. **Which risk tier?** → Low / Moderate / High at calibrated thresholds
4. **Deployed where?** → FastAPI REST endpoint, JSON in / JSON out

---

## Quick Results

| Model | ROC-AUC | AUPRC | Notes |
|---|---:|---:|---|
| XGBoost, 74 features | 0.6797 | 0.2259 | Best AUC; isotonic calibration |
| LightGBM, 65 features, Optuna | 0.6781 | 0.2272 | Best AUPRC original run |
| LightGBM, 74 features, Optuna | 0.6765 | 0.2246 | Expanded feature set |
| **LightGBM, Optuna re-tune (MCC)** | **0.6767** | **0.2286** | **Best final model; saved to models/** |

74 engineered features from 50 raw fields. LightGBM Optuna re-tune leads on AUPRC — the operationally correct metric at 11.2% positive rate. Published literature ceiling for structured-only EHR readmission is ~0.73 AUC.

Production artifact: `models/readmission_xgb.joblib`.

Top 5 SHAP drivers:
- `discharge_disposition_id`
- `number_inpatient`
- `inpatient_ratio`
- `diag_1_cat`
- `prior_visits`

Generated explainability artifacts live in `reports/figures/`: `shap_summary.png`, `shap_waterfall_high_risk.png`, `shap_waterfall_low_risk.png`, and `calibration_curve.png`.

## Dataset

**Diabetes 130-US Hospitals (1999–2008)** — UCI ML Repository / Kaggle  
101,766 real patient encounters across 130 US hospitals. 50 features including demographics, ICD-9 diagnosis codes, lab results, medications, and prior visit history.

| Split | Count |
|-------|-------|
| Total encounters | 101,766 |
| Readmitted < 30 days (positive) | 11,357 (11.2%) |
| Readmitted > 30 days | 35,545 |
| Not readmitted | 54,864 |

---

## Feature Engineering

Raw clinical data required significant transformation before modeling:

| Engineering step | Detail |
|---|---|
| ICD-9 categorization | Maps raw diagnosis codes to 9 clinical groups (circulatory, respiratory, diabetes, injury, etc.) |
| Medication activity | 23 diabetes medications → `n_meds_active`, `n_meds_changed`, `n_meds_up`, `n_meds_down` |
| Insulin flags | `insulin_active`, `insulin_changed` — insulin management is a strong readmission signal |
| Visit history | `prior_visits`, `inpatient_ratio`, `high_prior_inpatient` (≥2 prior inpatient stays) |
| Intensity ratios | `procedures_per_day`, `labs_per_day`, `meds_per_diagnosis` |
| Age normalization | Age brackets → numeric midpoints |
| Missing data | Weight (97% missing) dropped; `?` values treated as unknown category |

---

## Model

**XGBoost** with 5-fold stratified CV. Class imbalance handled via `scale_pos_weight=4`.

| Metric | XGBoost | LGB Optuna (MCC) |
|--------|-------|-------|
| OOF ROC-AUC | 0.6797 | 0.6767 |
| OOF AUPRC | 0.2259 | 0.2286 |
| Brier score (calibrated) | ~0.090 | — |

OOF AUC of ~0.68 is consistent with published literature — readmission prediction from structured EHR data caps around 0.68–0.73 in peer-reviewed work. The positive class rate is 11.2%, making AUPRC the more informative metric operationally.

---

## SHAP Explainability

Every prediction comes with a per-patient SHAP breakdown of which features drove the score. Top global drivers:

- `number_inpatient` — prior inpatient visits (strongest signal)
- `time_in_hospital` — longer stays correlate with higher complexity
- `discharge_disposition_id` — where the patient went after discharge
- `n_meds_active` — breadth of active diabetes medication regimen
- `diag_1_cat` — primary diagnosis clinical category

---

## Deployment

```bash
uvicorn api.serve:app --host 0.0.0.0 --port 8000
```

**POST /predict** — JSON patient record in, risk score + SHAP factors out:

```json
{
  "readmit_prob_30d": 0.31,
  "risk_tier": "HIGH",
  "top_risk_factors": [
    {"feature": "number_inpatient", "shap": 0.48},
    {"feature": "time_in_hospital", "shap": 0.21},
    {"feature": "discharge_disposition_id", "shap": -0.18}
  ]
}
```

**GET /health** — liveness check.

---

## Quickstart

```bash
git clone https://github.com/danielbandy1/hospital-readmission
cd hospital-readmission
pip install -r requirements.txt
kaggle datasets download -d brandao/diabetes --unzip -p data/
python3 train_pipeline.py
uvicorn api.serve:app --port 8000
```

## VSCode Remote Responsiveness

If VSCode over SSH feels laggy while typing, this repo includes workspace settings at `.vscode/settings.json` to reduce file watching/indexing load by excluding large artifact folders (`models/`, `figures/`, `reports/figures/`, `notebooks/`).

---

## Stack

Python · XGBoost · LightGBM · SHAP · scikit-learn · FastAPI · pandas · Pydantic · pytest

## Author

Daniel Bandy — Mathematics & Statistics/Data Science, University of Kentucky  
[github.com/danielbandy1](https://github.com/danielbandy1) · dbandy134@outlook.com
