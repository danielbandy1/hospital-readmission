#!/usr/bin/env python3
"""
FastAPI inference endpoint — hospital 30-day readmission risk.

Run:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import shap

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "readmission_xgb.joblib"

app = FastAPI(
    title="Hospital 30-Day Readmission Risk",
    description="Predicts probability of a diabetic patient being readmitted within 30 days.",
    version="1.0.0",
)

_bundle  = None
_explainer = None


def _load():
    global _bundle, _explainer
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_bundle["model"])


class PatientRecord(BaseModel):
    race: str = Field("Caucasian", description="Patient race")
    gender: str = Field("Female")
    age: str = Field("[50-60)", description="Age bracket e.g. '[50-60)'")
    admission_type_id: int = Field(1)
    discharge_disposition_id: int = Field(1)
    admission_source_id: int = Field(7)
    time_in_hospital: int = Field(3)
    medical_specialty: str = Field("InternalMedicine")
    num_lab_procedures: int = Field(40)
    num_procedures: int = Field(1)
    num_medications: int = Field(10)
    number_outpatient: int = Field(0)
    number_emergency: int = Field(0)
    number_inpatient: int = Field(0)
    diag_1: str = Field("250.01", description="Primary ICD-9 diagnosis")
    diag_2: str = Field("?")
    diag_3: str = Field("?")
    number_diagnoses: int = Field(5)
    max_glu_serum: str = Field("None")
    A1Cresult: str = Field("None")
    insulin: str = Field("No", description="No / Down / Steady / Up")
    change: str = Field("No", description="No / Ch")
    diabetesMed: str = Field("Yes", description="Yes / No")
    metformin: str = Field("No")
    glipizide: str = Field("No")
    glyburide: str = Field("No")
    pioglitazone: str = Field("No")
    rosiglitazone: str = Field("No")
    repaglinide: str = Field("No")
    nateglinide: str = Field("No")
    chlorpropamide: str = Field("No")
    glimepiride: str = Field("No")
    acetohexamide: str = Field("No")
    tolbutamide: str = Field("No")
    acarbose: str = Field("No")
    miglitol: str = Field("No")
    troglitazone: str = Field("No")
    tolazamide: str = Field("No")
    examide: str = Field("No")
    citoglipton: str = Field("No")
    glyburide_metformin: str = Field("No", alias="glyburide-metformin")
    glipizide_metformin: str = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: str = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: str = Field("No", alias="metformin-pioglitazone")

    class Config:
        populate_by_name = True


class RiskResponse(BaseModel):
    readmit_prob_30d: float
    risk_tier: str
    top_risk_factors: list[dict]


@app.on_event("startup")
def startup():
    _load()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/predict", response_model=RiskResponse)
def predict(record: PatientRecord):
    _load()
    model     = _bundle["model"]
    feat_cols = _bundle["feature_cols"]

    row = record.model_dump(by_alias=True)
    row["encounter_id"] = 0
    row["patient_nbr"]  = 0
    row["weight"] = "?"
    row["payer_code"] = "?"
    row["readmitted"] = "NO"

    from src.features import build_features
    df = pd.DataFrame([row])
    try:
        df = build_features(df)
        df.drop(columns=["readmitted"], errors="ignore", inplace=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature error: {e}")

    cat_cols = df.select_dtypes("category").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].cat.codes

    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feat_cols].astype(np.float32)

    prob = float(model.predict_proba(X)[0, 1])

    if prob >= 0.35:
        tier = "HIGH"
    elif prob >= 0.18:
        tier = "MODERATE"
    else:
        tier = "LOW"

    shap_vals = _explainer.shap_values(X)[0]
    factors = sorted(
        [{"feature": f, "shap": round(float(s), 4)} for f, s in zip(feat_cols, shap_vals)],
        key=lambda x: abs(x["shap"]),
        reverse=True,
    )[:5]

    return RiskResponse(
        readmit_prob_30d=round(prob, 4),
        risk_tier=tier,
        top_risk_factors=factors,
    )
