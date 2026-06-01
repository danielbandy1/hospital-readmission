import numpy as np
import pandas as pd

AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

DROP_COLS = ["encounter_id", "patient_nbr", "weight", "payer_code"]

# discharge_disposition_id values that mean the patient left alive to home/facility
# (vs. expired=11, hospice=13/14) — useful as a direct predictor of return visits
_DISCHARGE_EXPIRED = {11, 19, 20, 21}  # expired / hospice
_DISCHARGE_HOME    = {1, 6, 8}         # home / home health / AMA
_DISCHARGE_FACILITY = {2, 3, 4, 5, 9, 10, 12, 15, 16, 17, 22, 23, 24, 27, 28, 29}


def _icd9_category(code):
    """Map ICD-9 code string to a broad clinical category."""
    if pd.isna(code) or code == "?":
        return "unknown"
    code = str(code).strip()
    if code.startswith("E") or code.startswith("V"):
        return "external"
    try:
        n = float(code)
    except ValueError:
        return "other"
    if 390 <= n < 460 or n == 785:
        return "circulatory"
    if 460 <= n < 520 or n == 786:
        return "respiratory"
    if 520 <= n < 580 or n == 787:
        return "digestive"
    if 250 <= n < 251:
        return "diabetes"
    if 800 <= n < 1000:
        return "injury"
    if 710 <= n < 740:
        return "musculoskeletal"
    if 580 <= n < 630 or n == 788:
        return "genitourinary"
    if 140 <= n < 240:
        return "neoplasms"
    if 290 <= n < 320:
        return "mental"
    if 320 <= n < 390:
        return "neurological"
    if 240 <= n < 280 and not (250 <= n < 251):
        return "endocrine"
    return "other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    df = df.mask(df == "?")

    df["age_num"] = df["age"].map(AGE_MAP).astype(float)
    df.drop(columns=["age"], inplace=True)

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1, "Unknown/Invalid": np.nan})

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_cat"] = df[col].apply(_icd9_category)
        df.drop(columns=[col], inplace=True)

    med_active = df[MED_COLS].isin(["Down", "Steady", "Up"])
    med_changed = df[MED_COLS].isin(["Down", "Up"])
    med_up = df[MED_COLS] == "Up"
    med_down = df[MED_COLS] == "Down"

    df["n_meds_active"] = med_active.sum(axis=1)
    df["n_meds_changed"] = med_changed.sum(axis=1)
    df["n_meds_up"] = med_up.sum(axis=1)
    df["n_meds_down"] = med_down.sum(axis=1)
    df["insulin_active"] = (df["insulin"].isin(["Down", "Steady", "Up"])).astype(int)
    df["insulin_changed"] = (df["insulin"].isin(["Down", "Up"])).astype(int)

    for col in MED_COLS:
        df[col] = df[col].map({"No": 0, "Down": -1, "Steady": 1, "Up": 2}).fillna(0).astype(int)

    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

    df["prior_visits"] = df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    df["inpatient_ratio"] = df["number_inpatient"] / (df["prior_visits"] + 1)
    df["high_prior_inpatient"] = (df["number_inpatient"] >= 2).astype(int)

    df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"].clip(lower=1)
    df["labs_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"].clip(lower=1)
    df["meds_per_diagnosis"] = df["num_medications"] / df["number_diagnoses"].clip(lower=1)

    # --- Clinical context features (previously unused) ---

    # Admission & discharge type (high-signal clinical variables)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "discharge_disposition_id" in df.columns:
        dd = df["discharge_disposition_id"].fillna(0).astype(int)
        df["discharge_to_home"]     = dd.isin(_DISCHARGE_HOME).astype(int)
        df["discharge_to_facility"] = dd.isin(_DISCHARGE_FACILITY).astype(int)
        df["discharge_expired"]     = dd.isin(_DISCHARGE_EXPIRED).astype(int)

    # A1c test result — strong readmission predictor for diabetics
    if "A1Cresult" in df.columns:
        df["a1c_tested"]    = (~df["A1Cresult"].isin(["None", np.nan])).astype(int)
        df["a1c_high"]      = df["A1Cresult"].isin([">7", ">8"]).astype(int)
        df["a1c_very_high"] = (df["A1Cresult"] == ">8").astype(int)
        df.drop(columns=["A1Cresult"], inplace=True)

    # Glucose serum measurement
    if "max_glu_serum" in df.columns:
        df["glu_tested"]    = (~df["max_glu_serum"].isin(["None", np.nan])).astype(int)
        df["glu_high"]      = df["max_glu_serum"].isin([">200", ">300"]).astype(int)
        df["glu_very_high"] = (df["max_glu_serum"] == ">300").astype(int)
        df.drop(columns=["max_glu_serum"], inplace=True)

    # Race (clinically relevant in diabetes outcomes literature)
    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown")

    # Medical specialty — binary: was there a specialist involved?
    if "medical_specialty" in df.columns:
        df["has_specialty"] = df["medical_specialty"].notna().astype(int)
        df.drop(columns=["medical_specialty"], inplace=True)

    # Interaction: insulin changed AND prior inpatient (high-risk signal)
    df["insulin_changed_x_prior_inpatient"] = (df["insulin_changed"] * df["number_inpatient"]).astype(float)

    # Emergency visit ratio — emergencies / all prior visits
    df["emergency_ratio"] = df["number_emergency"] / (df["prior_visits"] + 1)

    # Age-based risk flags — literature shows 70+ and <40 have distinct readmission profiles
    df["age_over_70"]  = (df["age_num"] >= 70).astype(int)
    df["age_under_40"] = (df["age_num"] < 40).astype(int)

    # Interaction: older patients + more prior inpatient = highest-risk cohort
    df["age_x_prior_inpatient"] = (df["age_num"] * df["number_inpatient"]).astype(float)

    # High-complexity stay flags
    df["long_stay"]           = (df["time_in_hospital"] > 7).astype(int)
    df["high_diag_burden"]    = (df["number_diagnoses"] >= 7).astype(int)
    df["multiple_emergency"]  = (df["number_emergency"] >= 2).astype(int)

    # Primary diagnosis is diabetes itself (direct metabolic admission)
    if "diag_1_cat" in df.columns:
        df["diag_diabetes_primary"] = (df["diag_1_cat"] == "diabetes").astype(int)

    # Medication churn score — active meds being adjusted simultaneously
    df["med_churn_score"] = (df["n_meds_active"] * df["n_meds_changed"]).astype(float)

    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    """Binary: readmitted within 30 days."""
    return (df["readmitted"] == "<30").astype(int)


FEATURE_COLS: list[str] = []
