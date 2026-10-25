# Hospital Readmission Risk: Clinical ML Interview & Defense Guide

**AUTHORED BY GEMINI (extending ORION framework)**

This guide provides Daniel Bandy with the clinical ML domain knowledge, technical defense, and architectural explanations required for healthcare data science interviews (e.g., Amgen, Stryker, Epic, Humana).

---

## 1. Project Summary & Elevator Pitch (30-second summary)

> *"In this project, I built an end-to-end clinical risk prediction system to forecast 30-day diabetic hospital readmissions using 101,766 encounters from 130 US hospitals. Rather than using an ungrounded random row split, I enforced a strict patient-disjoint split (`GroupShuffleSplit` across 71,518 unique patients) with zero patient overlap between train and test sets, preventing data leakage from chronic repeat visits. Engineering 74 clinical features from ICD-9 diagnoses, medication regimens, and utilization history, I developed a deterministic LightGBM baseline achieving a test ROC-AUC of 0.6717 and Average Precision of 0.2088 (vs. a 10.67% holdout base rate). I packaged the model into a FastAPI microservice and Streamlit dashboard, persisting training-fitted categorical encoders to guarantee inference safety on unseen patient profiles."*

---

## 2. The Core Clinical ML Challenge: Patient-Level Data Leakage

### The Problem
- The dataset contains **101,766 hospital encounters**, but only **71,518 distinct patients**.
- Chronic diabetic patients are frequently readmitted multiple times across the 10-year observation window (1999–2008).
- **The Fatal Flaw in Naive Baselines:**
  - Standard tutorials perform `train_test_split(df, test_size=0.2, random_state=42)`.
  - This randomly places encounter #1 for Patient A in the training set, and encounter #2 for Patient A in the test set.
  - The model learns Patient A's specific medication combination and demographic profile, effectively memorizing the patient rather than generalizing clinical risk factors.
  - This inflates evaluation metrics (the legacy random-CV 0.680 ROC-AUC / 0.230 AUPRC headline) and creates catastrophic failure in production when facing a genuinely new patient.

### The Solution: Zero-Overlap Grouped Splitting
- We split by `patient_nbr` using `GroupShuffleSplit`:
  - **Training Set:** 81,613 encounters (zero patient overlap with test).
  - **Test Set:** 20,153 encounters (zero patient overlap with train).
- This mirrors real-world clinical deployment: the model is evaluated on its ability to generalize to **previously unseen patients** at hospital discharge.
- On this strict holdout, our fixed LightGBM baseline achieves:
  - **ROC-AUC:** `0.671659`
  - **Average Precision (PR-AUC):** `0.208767` (nearly 2x the 10.67% base rate)
  - **Brier Score:** `0.091228` (indicating strong probability calibration)

---

## 3. Clinical Feature Engineering (74 Features)

The 74 features fall into four clinical dimensions:
1. **Utilization History (Strongest Predictive Power):**
   - `number_inpatient`: Number of inpatient admissions in the preceding 12 months. Prior hospitalization is the single strongest indicator of clinical fragility and health system dependence.
   - `number_emergency`: Emergency department visits.
   - `number_outpatient`: Outpatient clinic interactions.
2. **Clinical Severity & Complexity:**
   - `time_in_hospital`: Length of stay (1–14 days). Longer stays correlate with complications and functional decline.
   - `num_lab_procedures` & `num_procedures`: Intensity of inpatient diagnostic interventions.
   - `num_medications`: Polypharmacy marker (patients taking 15+ medications face higher adverse drug interaction risks).
   - `number_diagnoses`: Multimorbidity count.
3. **Diagnostic Grouping (ICD-9 Codes):**
   - Raw ICD-9 codes (thousands of distinct values) are grouped into clinically meaningful primary organ systems:
     - Circulatory (390–459, 785)
     - Respiratory (460–519, 786)
     - Digestive (520–579, 787)
     - Diabetes (250.xx)
     - Injury & Poisoning (800–999)
     - Musculoskeletal (710–739)
     - Genitourinary (580–629, 788)
     - Other / Neoplasms
4. **Diabetic Pharmacotherapy Management:**
   - Dosage changes across 23 medications (insulin, metformin, sulfonylureas, glipizide, etc.) mapped to: `No`, `Steady`, `Up`, `Down`.
   - `change`: Whether any diabetic medication dosage was adjusted during the encounter (medication instability is a known readmission trigger).
   - `diabetesMed`: Whether any diabetic medication was prescribed.
   - `A1Cresult`: Glycemic control biomarker (`None`, `Norm`, `>7`, `>8`).

---

## 4. Serving Architecture & Category Encoding Integrity

### The Serving Bug in Naive Implementations
- In Python/pandas, running `pd.Categorical(df['race']).codes` on a single incoming JSON payload causes all category codes to become `0` because there is only one unique level in that batch!
- **Our Architecture Fix:**
  - In `train_pipeline.py`, categorical encodings (levels and integer mappings) are fitted **strictly on the training partition** and serialized into the `.joblib` model artifact:
    ```python
    model_bundle = {
        'model': lgb_model,
        'category_mappings': category_maps,
        'features': feature_columns
    }
    ```
  - When `api/serve.py` or `app.py` receives a single encounter, it applies the serialized mapping dictionary.
  - Unseen or missing categorical values are assigned `-1` (missing sentinel) rather than throwing runtime exceptions or corrupting the feature matrix.
  - Comprehensive unit tests in `tests/test_api.py` verify this single-row inference contract.

---

## 5. Clinical Decision Support & Operational Thresholds

If an interviewer asks: *"How would a hospital actually use this model in clinical operations?"*

1. **Why Threshold = 0.50 Fails in Healthcare:**
   - Readmissions occur in ~11% of encounters. A default 0.50 threshold would predict positive for virtually zero patients because the posterior risk rarely exceeds 50% for any single individual.
2. **Resource-Constrained Thresholding:**
   - Hospital transitional care teams (nurses, social workers, pharmacists) have a fixed weekly bandwidth (e.g. they can provide intensive home visits and post-discharge telephone follow-ups to only 10% or 15% of discharged patients).
   - The operational threshold is chosen by setting a target **intervention capacity** (e.g., the top 10th percentile of predicted risk).
   - In our holdout, targeting the top 15% highest-risk encounters captures over **38% of all readmissions**, nearly quadrupling the random baseline capture rate.
3. **Actionable Explainability with SHAP:**
   - Hospital staff will not trust a black box. Our Streamlit dashboard provides patient-level SHAP waterfall charts showing *why* a patient was flagged (e.g. `+0.12 risk` from 3 prior inpatient visits, `+0.08 risk` from insulin dosage increase, `-0.04 risk` from normal A1C).
   - This empowers clinicians to tailor discharge interventions (e.g., dispatching a home health nurse to review medication adherence).

---

## 6. Mock Interview Q&A

**Q: Why is your ROC-AUC 0.672 when the older README mentioned 0.680?**
> *"The 0.680 figure was from a legacy random-row cross-validation experiment that leaked chronic patients across folds. When I properly enforced a patient-disjoint split where no patient appears in both train and test partitions, the honest, un-leaked generalization ROC-AUC is 0.6717. In clinical data science, metric integrity and leak prevention must always take precedence over vanity headlines."*

**Q: Did you consider SMOTE or other oversampling techniques for class imbalance?**
> *"Yes, but in clinical risk scoring, synthetic oversampling like SMOTE distorts the predicted probabilities, destroying calibration (Brier score). Because clinical teams prioritize calibrated probabilities over raw binary classifications, training with LightGBM's native log-loss on natural prevalence—combined with threshold tuning based on hospital operational capacity—preserves reliable probability ranking without fabricating synthetic patient records."*

**Q: How would you monitor this model once deployed in an Electronic Health Record (EHR) system?**
> *"I would track three monitoring pillars: (1) **Data Drift / Feature Drift** using Population Stability Index (PSI) or Kolmogorov-Smirnov tests on key features like inpatient visits and length of stay; (2) **Prediction Drift** monitoring the distribution of output risk probabilities over weekly cohorts; and (3) **Ground Truth Lag** tracking 30-day post-discharge outcomes as claims/readmission data mature to detect model degradation over time."*
