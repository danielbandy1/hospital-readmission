#!/usr/bin/env python3
"""
Streamlit dashboard — Hospital 30-Day Readmission Risk Scorer.

Run:
    streamlit run app.py
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import build_features

MODEL_DIR = pathlib.Path("models")
MODEL_CANDIDATES = [
    MODEL_DIR / "readmission_lgb.joblib",
    MODEL_DIR / "readmission_xgb.joblib",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[-1])

st.set_page_config(
    page_title="Readmission Risk Scorer",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    try:
        import shap
        explainer = shap.TreeExplainer(bundle["model"])
    except Exception:
        explainer = None
    return bundle, explainer


def _tier_color(tier):
    return {"LOW": "#2ecc71", "MODERATE": "#f39c12", "HIGH": "#e74c3c"}.get(tier, "#888")


def _predict(bundle, explainer, row_dict):
    row = {**row_dict, "encounter_id": 0, "patient_nbr": 0,
           "weight": "?", "payer_code": "?", "readmitted": "NO"}
    df = pd.DataFrame([row])
    df = build_features(df)
    df.drop(columns=["readmitted"], errors="ignore", inplace=True)
    for col in df.select_dtypes("category").columns:
        df[col] = df[col].cat.codes
    feat_cols = bundle["feature_cols"]
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feat_cols].astype(np.float32)

    prob = float(bundle["model"].predict_proba(X)[0, 1])
    if "calibrator" in bundle:
        prob = float(bundle["calibrator"].predict([prob])[0])

    tier = "HIGH" if prob >= 0.35 else ("MODERATE" if prob >= 0.18 else "LOW")

    shap_vals = None
    if explainer is not None:
        try:
            sv = explainer.shap_values(X)
            arr = np.array(sv[1] if isinstance(sv, list) else sv)[0]
            shap_vals = sorted(
                zip(feat_cols, arr), key=lambda x: abs(x[1]), reverse=True
            )[:10]
        except Exception:
            pass

    return prob, tier, shap_vals, X


# ── Sidebar: patient inputs ──────────────────────────────────────────────────

st.sidebar.header("Patient Record")

age = st.sidebar.selectbox("Age bracket", [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
], index=5)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

time_in_hospital = st.sidebar.slider("Days in hospital", 1, 14, 4)

number_inpatient = st.sidebar.slider("Prior inpatient visits", 0, 10, 0)
number_outpatient = st.sidebar.slider("Prior outpatient visits", 0, 10, 0)
number_emergency = st.sidebar.slider("Prior emergency visits", 0, 10, 0)

num_medications = st.sidebar.slider("# medications", 1, 40, 12)
num_procedures = st.sidebar.slider("# procedures", 0, 6, 1)
num_lab_procedures = st.sidebar.slider("# lab procedures", 1, 120, 44)
number_diagnoses = st.sidebar.slider("# diagnoses", 1, 16, 7)

diag_1 = st.sidebar.text_input("Primary ICD-9 diagnosis", value="250.01")
diag_2 = st.sidebar.text_input("Secondary ICD-9 diagnosis", value="?")
diag_3 = st.sidebar.text_input("Tertiary ICD-9 diagnosis", value="?")

admission_type_id = st.sidebar.selectbox(
    "Admission type", [1, 2, 3, 4, 5, 6, 7, 8],
    format_func=lambda x: {1:"Emergency",2:"Urgent",3:"Elective",4:"Newborn",
                            5:"Not available",6:"NULL",7:"Trauma center",8:"Not mapped"}.get(x,str(x))
)
discharge_disposition_id = st.sidebar.selectbox(
    "Discharge to", [1, 2, 3, 4, 6, 11, 13, 14],
    format_func=lambda x: {1:"Home",2:"SNF",3:"Skilled nursing",4:"ICF",
                            6:"Home health care",11:"Expired",13:"Hospice/home",14:"Hospice/facility"}.get(x,str(x))
)
admission_source_id = st.sidebar.selectbox(
    "Admission source", [1, 4, 7, 17],
    format_func=lambda x: {1:"Physician referral",4:"Transfer",7:"Emergency room",17:"Walk-in"}.get(x,str(x)),
    index=2,
)

insulin = st.sidebar.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
change = st.sidebar.selectbox("Medication change?", ["No", "Ch"])
diabetesMed = st.sidebar.selectbox("On diabetes medication?", ["Yes", "No"])
A1Cresult = st.sidebar.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
max_glu_serum = st.sidebar.selectbox("Max glucose serum", ["None", "Norm", ">200", ">300"])
medical_specialty = st.sidebar.selectbox(
    "Medical specialty", ["InternalMedicine", "Family/GeneralPractice",
                           "Cardiology", "Surgery-General", "Orthopedics", "?"]
)

MED_DEFAULTS = {
    "metformin": "No", "repaglinide": "No", "nateglinide": "No",
    "chlorpropamide": "No", "glimepiride": "No", "acetohexamide": "No",
    "glipizide": "No", "glyburide": "No", "tolbutamide": "No",
    "pioglitazone": "No", "rosiglitazone": "No", "acarbose": "No",
    "miglitol": "No", "troglitazone": "No", "tolazamide": "No",
    "examide": "No", "citoglipton": "No",
    "glyburide-metformin": "No", "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No", "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
}

row_dict = {
    "age": age, "gender": gender, "race": "Caucasian",
    "time_in_hospital": time_in_hospital,
    "number_inpatient": number_inpatient,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "num_medications": num_medications,
    "num_procedures": num_procedures,
    "num_lab_procedures": num_lab_procedures,
    "number_diagnoses": number_diagnoses,
    "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
    "admission_type_id": admission_type_id,
    "discharge_disposition_id": discharge_disposition_id,
    "admission_source_id": admission_source_id,
    "insulin": insulin, "change": change, "diabetesMed": diabetesMed,
    "A1Cresult": A1Cresult, "max_glu_serum": max_glu_serum,
    "medical_specialty": medical_specialty,
    **MED_DEFAULTS,
}

# ── Main panel ───────────────────────────────────────────────────────────────

st.title("🏥 30-Day Readmission Risk Scorer")
model_label = "LightGBM (tuned)" if "lgb" in str(MODEL_PATH) else "XGBoost"
st.caption(f"Model: **{model_label}** · Dataset: 101,766 diabetic encounters · UCI 130-Hospitals")

bundle, explainer = load_model()
prob, tier, shap_vals, X = _predict(bundle, explainer, row_dict)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.metric("Readmission probability", f"{prob*100:.1f}%")

with col2:
    color = _tier_color(tier)
    st.markdown(
        f"<div style='background:{color};padding:12px 18px;border-radius:8px;"
        f"color:white;font-size:1.4rem;font-weight:700;text-align:center'>"
        f"{tier} RISK</div>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        | Risk tier | Threshold | Interpretation |
        |-----------|-----------|----------------|
        | 🟢 LOW | < 18% | Routine discharge planning |
        | 🟡 MODERATE | 18–35% | Consider follow-up call within 72h |
        | 🔴 HIGH | ≥ 35% | Care coordination recommended |
        """
    )

st.divider()

if shap_vals:
    import matplotlib.pyplot as plt

    st.subheader("Top drivers for this patient")
    features = [f for f, _ in shap_vals]
    values = [v for _, v in shap_vals]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on readmission probability)")
    ax.set_title("Feature contributions — red = increases risk, green = decreases risk")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("SHAP explainability not available for this model artifact.")

st.divider()

with st.expander("Population context (dataset baseline)"):
    st.markdown("""
    | Metric | Dataset value |
    |--------|--------------|
    | Overall readmission rate (<30d) | **11.2%** |
    | Median time in hospital | 4 days |
    | Median prior inpatient visits | 0 |
    | Most common primary diagnosis | Circulatory |
    """)
    st.markdown(
        "OOF AUC ~0.68–0.70 on this task; literature reports up to 0.73 with richer feature sets. "
        "Readmission prediction from structured EHR data alone is a genuinely hard problem."
    )

with st.expander("Model info"):
    cal_note = "Isotonic regression on OOF predictions" if "calibrator" in bundle else "None (raw model probabilities)"
    st.markdown(f"""
    - **Model file:** `{MODEL_PATH.name}`
    - **Features:** {len(bundle['feature_cols'])}
    - **OOF AUPRC:** ~0.227  |  **OOF AUC:** ~0.680
    - **Positive class rate:** 11.2% (class imbalance handled via `scale_pos_weight`)
    - **Calibration:** {cal_note}
    - **Explainability:** TreeSHAP (Lundberg & Lee, 2017)
    """)
