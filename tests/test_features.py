import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.features import AGE_MAP, MED_COLS, _icd9_category, build_features, make_target


def _row(**overrides):
    base = {
        "encounter_id": 1,
        "patient_nbr": 100,
        "weight": "?",
        "payer_code": "?",
        "race": "Caucasian",
        "age": "[50-60)",
        "gender": "Female",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "250.01",
        "diag_2": "401",
        "diag_3": "?",
        "number_diagnoses": 5,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
        "readmitted": "NO",
        **{med: "No" for med in MED_COLS},
    }
    base.update(overrides)
    return base


@pytest.fixture
def single_df():
    return pd.DataFrame([_row()])


# --- ICD-9 category ---

@pytest.mark.parametrize("code,expected", [
    ("410",    "circulatory"),
    ("480",    "respiratory"),
    ("530",    "digestive"),
    ("250.01", "diabetes"),
    ("810",    "injury"),
    ("715",    "musculoskeletal"),
    ("590",    "genitourinary"),
    ("200",    "neoplasms"),
    ("260",    "endocrine"),
    ("E890",   "external"),
    ("V10",    "external"),
    ("?",      "unknown"),
    (None,     "unknown"),
    ("xyz",    "other"),
])
def test_icd9_category(code, expected):
    assert _icd9_category(code) == expected


# --- make_target ---

def test_make_target():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})
    assert list(make_target(df)) == [1, 0, 0, 1]


# --- build_features: column cleanup ---

def test_drop_cols_removed(single_df):
    out = build_features(single_df)
    for col in ["encounter_id", "patient_nbr", "weight", "payer_code"]:
        assert col not in out.columns


def test_age_bracket_replaced(single_df):
    out = build_features(single_df)
    assert "age_num" in out.columns
    assert "age" not in out.columns
    assert out["age_num"].iloc[0] == 55.0


def test_all_age_brackets_map():
    rows = [_row(age=b) for b in AGE_MAP]
    out = build_features(pd.DataFrame(rows))
    assert out["age_num"].isna().sum() == 0


def test_diag_columns_replaced(single_df):
    out = build_features(single_df)
    for col in ["diag_1_cat", "diag_2_cat", "diag_3_cat"]:
        assert col in out.columns
    for col in ["diag_1", "diag_2", "diag_3"]:
        assert col not in out.columns
    assert out["diag_1_cat"].iloc[0] == "diabetes"
    assert out["diag_2_cat"].iloc[0] == "circulatory"


# --- build_features: medication aggregates ---

def test_med_aggregates_all_no(single_df):
    out = build_features(single_df)
    assert out["n_meds_active"].iloc[0] == 0
    assert out["n_meds_changed"].iloc[0] == 0
    assert out["n_meds_up"].iloc[0] == 0
    assert out["n_meds_down"].iloc[0] == 0


def test_med_aggregates_with_active():
    out = build_features(pd.DataFrame([_row(insulin="Steady", metformin="Up", glipizide="Down")]))
    assert out["n_meds_active"].iloc[0] == 3
    assert out["n_meds_changed"].iloc[0] == 2
    assert out["n_meds_up"].iloc[0] == 1
    assert out["n_meds_down"].iloc[0] == 1
    assert out["insulin_active"].iloc[0] == 1
    assert out["insulin_changed"].iloc[0] == 0


def test_insulin_changed_flag():
    out = build_features(pd.DataFrame([_row(insulin="Up")]))
    assert out["insulin_active"].iloc[0] == 1
    assert out["insulin_changed"].iloc[0] == 1


# --- build_features: derived ratios ---

def test_derived_ratios_present(single_df):
    out = build_features(single_df)
    for col in ["procedures_per_day", "labs_per_day", "meds_per_diagnosis",
                "prior_visits", "inpatient_ratio", "high_prior_inpatient"]:
        assert col in out.columns


def test_prior_visits_sum(single_df):
    out = build_features(single_df)
    assert out["prior_visits"].iloc[0] == 0


def test_high_prior_inpatient_flag():
    low  = build_features(pd.DataFrame([_row(number_inpatient=1)]))
    high = build_features(pd.DataFrame([_row(number_inpatient=2)]))
    assert low["high_prior_inpatient"].iloc[0] == 0
    assert high["high_prior_inpatient"].iloc[0] == 1


# --- build_features: edge cases ---

def test_question_marks_handled():
    out = build_features(pd.DataFrame([_row(diag_1="?", diag_2="?", diag_3="?")]))
    assert out["diag_1_cat"].iloc[0] == "unknown"


def test_unknown_gender_is_nan():
    out = build_features(pd.DataFrame([_row(gender="Unknown/Invalid")]))
    assert pd.isna(out["gender"].iloc[0])


def test_output_row_count(single_df):
    out = build_features(single_df)
    assert len(out) == 1


def test_output_has_enough_features(single_df):
    out = build_features(single_df)
    assert out.shape[1] >= 20
