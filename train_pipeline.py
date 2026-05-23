#!/usr/bin/env python3
"""
Hospital 30-day readmission prediction pipeline.

Usage:
    python3 train_pipeline.py
    python3 train_pipeline.py --folds 10 --no-shap
"""
import argparse
import pathlib
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.features import build_features, make_target
from src.evaluate import print_metrics, plot_roc_pr

DATA_PATH  = pathlib.Path("data/diabetic_data.csv")
MODEL_DIR  = pathlib.Path("models")
FIGURE_DIR = pathlib.Path("figures")

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 10,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 4.0,
    "n_estimators":     1000,
    "early_stopping_rounds": 50,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",   type=int,  default=5)
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    print("Loading data...", flush=True)
    raw = pd.read_csv(DATA_PATH)
    print(f"  {raw.shape[0]:,} rows × {raw.shape[1]} columns", flush=True)

    y = make_target(raw)
    print(f"  Positive class (<30 readmit): {y.mean()*100:.1f}%  ({y.sum():,} / {len(y):,})", flush=True)

    print("Engineering features...", flush=True)
    df = build_features(raw)
    df.drop(columns=["readmitted"], inplace=True)

    cat_cols = df.select_dtypes("category").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].cat.codes

    X = df.astype(np.float32)
    feature_cols = X.columns.tolist()
    print(f"  {len(feature_cols)} features", flush=True)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    fold_aucs  = []
    models     = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        prob = model.predict_proba(X_va)[:, 1]
        auc  = roc_auc_score(y_va, prob)
        fold_aucs.append(auc)
        oof_probs[va_idx] = prob
        models.append(model)
        print(f"  Fold {fold}/{args.folds}  AUC={auc:.4f}  trees={model.best_iteration}", flush=True)

    oof_auc = roc_auc_score(y, oof_probs)
    oof_ap  = average_precision_score(y, oof_probs)
    print(f"\nOOF ROC-AUC : {oof_auc:.4f}  (std={np.std(fold_aucs):.4f})")
    print(f"OOF Avg Prec: {oof_ap:.4f}")

    print("\nFull classification report (threshold=0.5):")
    print_metrics(y.values, oof_probs)

    plot_roc_pr(y.values, oof_probs, save_path=FIGURE_DIR / "roc_pr_curves.png")
    print(f"Curves saved → {FIGURE_DIR}/roc_pr_curves.png")

    print("\nRetraining on full data...", flush=True)
    params_full = {k: v for k, v in XGB_PARAMS.items()
                   if k not in ("early_stopping_rounds",)}
    params_full["n_estimators"] = int(np.mean([m.best_iteration for m in models]))
    final_model = xgb.XGBClassifier(**params_full)
    final_model.fit(X, y, verbose=False)

    model_path = MODEL_DIR / "readmission_xgb.joblib"
    joblib.dump({"model": final_model, "feature_cols": feature_cols}, model_path)
    print(f"Model saved → {model_path}")

    if not args.no_shap:
        print("\nComputing SHAP values (sample of 5000)...", flush=True)
        sample_idx = np.random.default_rng(42).choice(len(X), size=min(5000, len(X)), replace=False)
        X_sample = X.iloc[sample_idx]
        explainer = shap.TreeExplainer(final_model)
        shap_vals = explainer.shap_values(X_sample)

        plt_shap = shap.summary_plot(shap_vals, X_sample, show=False, max_display=20)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary saved → {FIGURE_DIR}/shap_summary.png")

        imp = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        print("\nTop 15 features by SHAP:")
        print(imp.head(15).to_string(index=False))
        imp.to_csv(FIGURE_DIR / "feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
