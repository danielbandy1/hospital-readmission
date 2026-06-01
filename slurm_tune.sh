#!/bin/bash
#SBATCH --job-name=hosp_lgb_tune
#SBATCH --account=coa_ich248_uksr
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dbandy134@outlook.com

set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models figures

echo "Job ID: $SLURM_JOB_ID"
echo "Host:   $(hostname)"
echo "Start:  $(date)"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    module load python 2>/dev/null || true
    python3 -m venv venv
    source venv/bin/activate
    pip install --quiet lightgbm optuna scikit-learn pandas numpy joblib matplotlib seaborn
fi

echo "Python: $(which python3)"
echo "LightGBM: $(python3 -c 'import lightgbm; print(lightgbm.__version__)')"
echo "Optuna:   $(python3 -c 'import optuna; print(optuna.__version__)')"

python3 tune_lgb.py \
    --trials 150 \
    --folds 10 \
    --retrain \
    --jobs 1

echo "Done: $(date)"
