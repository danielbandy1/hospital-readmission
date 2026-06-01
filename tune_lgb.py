#!/usr/bin/env python3
"""
LightGBM + Optuna hyperparameter tuning for 30-day readmission.

Objective: maximize OOF AUPRC (better than AUC at 11% positive rate).
Saves best params to models/best_lgb_params.json and final OOF scores
to models/lgb_results.json after an optional final retrain.

Usage:
    python3 tune_lgb.py                          # 100 trials, 5 folds
    python3 tune_lgb.py --trials 150 --folds 10  # MCC overnight run
    python3 tune_lgb.py --retrain                # after tuning, fit final model
"""
import argparse
import json
import pathlib
import warnings
warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.features import build_features, make_target

DATA_PATH  = pathlib.Path("data/diabetic_data.csv")
MODEL_DIR  = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_xy():
    raw = pd.read_csv(DATA_PATH)
    y   = make_target(raw)
    df  = build_features(raw)
    df.drop(columns=["readmitted"], inplace=True)
    cat_cols = df.select_dtypes("category").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].cat.codes
    X = df.astype(np.float32)
    return X, y


def _objective(trial, X, y, n_folds):
    params = {
        "objective":         "binary",
        "metric":            "average_precision",
        "verbosity":         -1,
        "learning_rate":     trial.suggest_float("learning_rate",    0.01, 0.3,  log=True),
        "num_leaves":        trial.suggest_int  ("num_leaves",       20,   400),
        "max_depth":         trial.suggest_int  ("max_depth",        3,    10),
        "min_child_samples": trial.suggest_int  ("min_child_samples",10,   150),
        "subsample":         trial.suggest_float("subsample",        0.5,  1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5,  1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha",        1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",       1e-4, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain",   0.0,  1.0),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0,  12.0),
        "n_jobs":            -1,
        "random_state":      42,
    }
    n_estimators = trial.suggest_int("n_estimators", 200, 3000)

    cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof     = np.zeros(len(y))
    pruning = optuna.integration.LightGBMPruningCallback(trial, "average_precision", valid_name="valid_0")

    for step, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(
            **params,
            n_estimators=n_estimators,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
                pruning,
            ],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return average_precision_score(y, oof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int,  default=100)
    parser.add_argument("--folds",   type=int,  default=5)
    parser.add_argument("--retrain", action="store_true", help="Fit final model on all data after tuning")
    parser.add_argument("--jobs",    type=int,  default=1,  help="Optuna parallel jobs (1 for SLURM, >1 for local)")
    args = parser.parse_args()

    print(f"Loading data...", flush=True)
    X, y = _load_xy()
    print(f"  {len(X):,} rows  {X.shape[1]} features  {y.mean()*100:.1f}% positive", flush=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: _objective(trial, X, y, args.folds),
        n_trials=args.trials,
        n_jobs=args.jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n--- Optuna complete ---")
    print(f"Best OOF AUPRC : {best.value:.4f}")
    print(f"Best params    : {json.dumps(best.params, indent=2)}")

    params_path = MODEL_DIR / "best_lgb_params.json"
    with open(params_path, "w") as f:
        json.dump({"auprc": best.value, "params": best.params}, f, indent=2)
    print(f"Saved → {params_path}")

    if args.retrain:
        print("\nFitting final model on full data...", flush=True)
        p = best.params.copy()
        n_est = p.pop("n_estimators")
        final = lgb.LGBMClassifier(
            objective="binary",
            verbosity=-1,
            n_estimators=n_est,
            n_jobs=-1,
            random_state=42,
            **p,
        )
        final.fit(X, y)

        cv      = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        oof     = np.zeros(len(y))
        for tr_idx, va_idx in cv.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            m = lgb.LGBMClassifier(
                objective="binary", verbosity=-1,
                n_estimators=n_est, n_jobs=-1, random_state=42, **p,
            )
            m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=-1)])
            oof[va_idx] = m.predict_proba(X_va)[:, 1]

        oof_auprc = average_precision_score(y, oof)
        oof_auc   = roc_auc_score(y, oof)
        print(f"OOF AUPRC: {oof_auprc:.4f}  |  OOF AUC: {oof_auc:.4f}")

        model_path = MODEL_DIR / "readmission_lgb.joblib"
        joblib.dump({"model": final, "feature_cols": X.columns.tolist()}, model_path)
        print(f"Model saved → {model_path}")

        results = {
            "model": "LightGBM (Optuna tuned)",
            "oof_auprc": round(oof_auprc, 4),
            "oof_auc":   round(oof_auc, 4),
            "n_trials":  args.trials,
            "n_folds":   args.folds,
            "best_params": best.params,
        }
        results_path = MODEL_DIR / "lgb_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
