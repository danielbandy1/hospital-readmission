import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, roc_curve, precision_recall_curve,
)
import matplotlib.pyplot as plt


def print_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Avg Prec: {ap:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Not <30", "Readmit <30"]))
    return auc, ap


def plot_roc_pr(y_true, y_prob, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax2.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax2.axhline(y_true.mean(), color="r", linestyle="--", lw=1, label="Baseline")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
