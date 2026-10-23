"""Serving must reuse training codes. AUTHORED BY ORION."""
import pandas as pd
from src.encoding import fit_maps,encode

def test_single_record_keeps_training_category_code():
    training=pd.DataFrame({'race':['Asian','Caucasian','Other'],'value':[1,2,3]})
    mapping=fit_maps(training)
    one=encode(training.iloc[[1]],mapping)
    all_rows=encode(training,mapping)
    assert one.race.iloc[0]==all_rows.race.iloc[1]==1
    unseen=encode(pd.DataFrame({'race':['Unseen'],'value':[4]}),mapping)
    assert unseen.race.iloc[0]==-1

def test_api_rejects_legacy_category_mapping(monkeypatch):
    from api import serve
    from fastapi import HTTPException
    import pytest
    monkeypatch.setattr(serve,'_bundle',{'model':object(),'feature_cols':['race']})
    with pytest.raises(HTTPException,match='training category maps'):
        serve._predict_one(serve.PatientRecord())
