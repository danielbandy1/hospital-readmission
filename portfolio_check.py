"""Reproduce public UCI counts and a NEW patient-disjoint baseline, not old CV.
AUTHORED BY ORION. No credentials, no server, no clinical-deployment claim.
"""
from pathlib import Path
import hashlib,io,json,urllib.request,zipfile
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from src.features import build_features,make_target
from src.encoding import fit_maps,encode
ROOT=Path(__file__).resolve().parent
URL='https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip'
SHA='f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a'

def main():
    cache=ROOT/'data/source.zip';cache.parent.mkdir(exist_ok=True)
    if not cache.exists():
        data=urllib.request.urlopen(URL,timeout=60).read()
        if hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('Source changed; review first')
        cache.write_bytes(data)
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==SHA
    with zipfile.ZipFile(cache) as z: raw=pd.read_csv(z.open('diabetic_data.csv'),keep_default_na=False)
    assert len(raw)==101766 and raw.patient_nbr.nunique()==71518
    y=make_target(raw); assert y.sum()==11357
    features=build_features(raw).drop(columns='readmitted')
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(features,y,raw.patient_nbr))
    assert not set(raw.patient_nbr.iloc[tr]) & set(raw.patient_nbr.iloc[te])
    maps=fit_maps(features.iloc[tr]); X=encode(features,maps)
    model=LGBMClassifier(n_estimators=100,num_leaves=15,learning_rate=.05,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(X.iloc[tr],y.iloc[tr]); probs=model.predict_proba(X.iloc[te])[:,1]
    result=dict(source_url=URL,source_sha256=SHA,raw_encounters=len(raw),unique_patients=int(raw.patient_nbr.nunique()),positive_encounters=int(y.sum()),base_rate=float(y.mean()),engineered_features=X.shape[1],train_rows=len(tr),test_rows=len(te),patient_overlap=0,model='NEW LightGBM 100-tree patient-disjoint holdout; no tuning/calibration',seed=42,test_auc=float(roc_auc_score(y.iloc[te],probs)),test_average_precision=float(average_precision_score(y.iloc[te],probs)),test_base_rate=float(y.iloc[te].mean()),test_brier=float(brier_score_loss(y.iloc[te],probs)),legacy_headline_reproduced=False)
    out=ROOT/'portfolio_results';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    models=ROOT/'models';models.mkdir(exist_ok=True)
    joblib.dump({'model':model,'feature_cols':X.columns.tolist(),'categorical_maps':maps,'evaluation':result},models/'readmission_group_holdout.joblib')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay,PrecisionRecallDisplay
    fig,ax=plt.subplots(1,2,figsize=(10,4));RocCurveDisplay.from_predictions(y.iloc[te],probs,ax=ax[0]);PrecisionRecallDisplay.from_predictions(y.iloc[te],probs,ax=ax[1]);fig.suptitle('New patient-disjoint holdout; UCI benchmark, not clinical validation');fig.tight_layout();fig.savefig(out/'holdout_curves.png',dpi=130);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
