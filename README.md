# Hospital 30-Day Readmission Prediction

End-to-end ML system predicting whether a diabetic patient will be readmitted within 30 days of discharge — the core problem driving CMS's Hospital Readmissions Reduction Program (HRRP), which penalizes hospitals up to 3% of Medicare payments for excess readmissions.

## The Problem

Hospital readmission within 30 days is one of the most studied quality metrics in US healthcare. For diabetic patients specifically, readmission rates run 15–20% — and preventable readmissions cost the US healthcare system over $17 billion annually. Identifying high-risk patients at discharge allows care teams to intervene: follow-up calls, medication reconciliation, care coordination.

This project builds a production-ready readmission scoring system that answers four questions a clinical team actually asks:

1. **Who is likely to come back within 30 days?** → XGBoost classifier, ROC-AUC ~0.68
2. **Why?** → Per-patient SHAP explanations surfaced in the API response
3. **Which risk tier?** → Low / Moderate / High at calibrated thresholds
4. **Deployed where?** → FastAPI REST endpoint, JSON in / JSON out

---

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

| Metric | Value |
|--------|-------|
| OOF ROC-AUC | ~0.68 |
| OOF Average Precision | ~0.28 |

OOF AUC of 0.68 on this dataset is consistent with published literature — readmission prediction from structured EHR data is a genuinely hard problem. The positive class rate is 11%, making average precision the more informative metric operationally.

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

---

## Stack

Python · XGBoost · SHAP · scikit-learn · FastAPI · pandas · Pydantic

## Author

Daniel Bandy — Mathematics & Statistics/Data Science, University of Kentucky  
[github.com/danielbandy1](https://github.com/danielbandy1) · dbandy134@outlook.com
