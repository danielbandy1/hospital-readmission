import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "readmission_xgb.joblib"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="trained model not found — run train_pipeline.py first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.serve import app
    with TestClient(app) as c:
        yield c


BASELINE = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
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
    "insulin": "No",
    "change": "No",
    "diabetesMed": "Yes",
    "metformin": "No",
    "glipizide": "No",
    "glyburide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "tolbutamide": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
}


# --- /health ---

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# --- /predict: response structure ---

def test_predict_status(client):
    assert client.post("/predict", json=BASELINE).status_code == 200


def test_predict_schema(client):
    body = client.post("/predict", json=BASELINE).json()
    assert "readmit_prob_30d" in body
    assert "risk_tier" in body
    assert "top_risk_factors" in body


def test_predict_prob_in_range(client):
    prob = client.post("/predict", json=BASELINE).json()["readmit_prob_30d"]
    assert 0.0 <= prob <= 1.0


def test_predict_tier_valid(client):
    tier = client.post("/predict", json=BASELINE).json()["risk_tier"]
    assert tier in {"LOW", "MODERATE", "HIGH"}


def test_predict_shap_factors(client):
    factors = client.post("/predict", json=BASELINE).json()["top_risk_factors"]
    assert 1 <= len(factors) <= 5
    for f in factors:
        assert "feature" in f
        assert "shap" in f
        assert isinstance(f["shap"], float)


# --- /predict: model behaviour ---

def test_high_risk_scores_higher(client):
    """More prior inpatient stays and longer stay should raise the score."""
    low  = client.post("/predict", json={**BASELINE, "number_inpatient": 0, "time_in_hospital": 1}).json()
    high = client.post("/predict", json={**BASELINE, "number_inpatient": 3, "time_in_hospital": 14}).json()
    assert high["readmit_prob_30d"] > low["readmit_prob_30d"]


def test_high_risk_tier(client):
    payload = {**BASELINE, "number_inpatient": 5, "time_in_hospital": 14, "number_emergency": 3}
    tier = client.post("/predict", json=payload).json()["risk_tier"]
    assert tier in {"MODERATE", "HIGH"}


def test_deterministic(client):
    """Same input should always return the same probability."""
    p1 = client.post("/predict", json=BASELINE).json()["readmit_prob_30d"]
    p2 = client.post("/predict", json=BASELINE).json()["readmit_prob_30d"]
    assert p1 == p2


# --- /predict: input variants ---

def test_predict_with_active_insulin(client):
    payload = {**BASELINE, "insulin": "Steady", "change": "Ch"}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_unknown_diag(client):
    payload = {**BASELINE, "diag_1": "?", "diag_2": "?", "diag_3": "?"}
    assert client.post("/predict", json=payload).status_code == 200
