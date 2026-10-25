# Hospital readmission: reproducible public-benchmark baseline

AUTHORED BY ORION — local repairs for Daniel's review, 23 October 2026.

**Question:** can discharge-time clinical features rank 30-day readmissions on previously unseen patients? This is an educational UCI benchmark, not a production-ready or clinically validated system.

## One command from a clean checkout

```bash
bash reproduce.sh
```

See [INTERVIEW_STUDY_GUIDE.md](INTERVIEW_STUDY_GUIDE.md) for clinical domain context, patient-level data leakage defense, 74-feature breakdown, threshold selection for hospital intervention teams, and mock clinical ML interview Q&A.

Python 3.13 and Linux/aarch64 tested. The command creates a fresh `.venv`, installs exact pins from `requirements-portfolio.lock`, fetches the hash-pinned public UCI archive if absent, fits a small deterministic LightGBM baseline and runs feature/API regression tests. Threads are capped at two. It does not log in, publish, launch a server or train on private data. First installation/download needs internet; the data cache makes subsequent computation offline.

## Data and licence

Clore, Cios, DeShazo and Strack (2014), **Diabetes 130-US Hospitals for Years 1999–2008**, UCI, https://doi.org/10.24432/C5230J . Official metadata/licence: https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008 (CC BY 4.0). Public released records only; local source cache is ignored by Git. The exact source URL/SHA256 and aggregate counts are saved in `portfolio_results/metrics.json`. No row-level patient identifiers are printed or added to the repository.

## New independently runnable result

101,766 encounters, **71,518 distinct patients**, 11,357 positive encounters (11.16%); **74 engineered features**. GroupShuffleSplit (seed 42) puts 81,613 encounters in train and 20,153 in test with **zero patient overlap**. A fixed 100-tree LightGBM, without Optuna tuning or calibration, yields **test ROC-AUC 0.671659 and average precision 0.208767** on this one split (test base rate 10.67%). The complete preprocessing and model are regenerated locally; `portfolio_results/holdout_curves.png` shows the actual holdout curves.

Categories are fitted on training patients and saved in the model bundle. API and dashboard reuse those exact codes, including unknown-category handling. A single-row inference no longer makes every category code zero. Tests cover this and reject legacy model bundles without category maps.

## Historical claims: do not confuse with this run

The old résumé says **LightGBM ROC-AUC 0.680 / AUPRC 0.230**. The checked-in `models/lgb_results.json` instead reports **0.6765 / 0.2246**; the old README lists several different runs. These do not reproduce the exact headline. `reports/calibration_metrics.json` has still different metrics and does not identify an independent evaluation split. Historical artifacts remain untouched for inspection, with the old README in `docs/README_UPSTREAM_20261023.md`; they are not endorsed as patient-disjoint test results. No saved OOF predictions permit independent reconstruction of those old scores.

`train_pipeline.py`/`tune_lgb.py` are historical exploratory random-row CV paths. They do not implement the new patient-disjoint benchmark; their in-sample calibration evaluation is not independent validation. Use the default reproduction command for résumé numbers. Repeated model selection on a fixed holdout can overfit it: this audit made no search over the new test split.

## Serving and limitations

`api/serve.py` and `app.py` use the locally regenerated `models/readmission_group_holdout.joblib` when present. FastAPI requests are tested in-process; no public endpoint was deployed. Streamlit is a local demo. Heuristic LOW/MODERATE/HIGH thresholds are not validated clinical action thresholds; SHAP contributions explain raw model output, not a causal effect. The model has no external hospital/time validation, fairness certification, independent calibration test, intervention study or clinical approval. Encounters include repeat patients; the split accounts for that within this extract, not all future distribution shift.

This branch's new code/documentation is ORION-assisted; Daniel must review and explain it before claiming independent authorship or proficiency.
