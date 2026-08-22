"""
Shared calibration and threshold-sweep utilities. Reused by Phase 2 script
and later by the fairness audit (Phase 7) and API layer (Phase 5).
"""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.evaluation.metrics import specificity_score
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_calibration(y_true, y_proba, n_bins: int = 10) -> dict:
    """Reliability curve + calibration error (mean |observed - predicted| per bin)."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))
    brier = float(brier_score_loss(y_true, y_proba))
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "mean_calibration_error": calibration_error,
        "brier_score": brier,
    }


def sweep_thresholds(y_true, y_proba, thresholds=None) -> list:
    """Precision/recall/specificity/F1 at each threshold. Returns list of dicts."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.71, 0.05), 2)

    results = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })
    return results


def pick_best_threshold(sweep_results: list, baseline_threshold: float = 0.5, min_precision: float = 0.0) -> dict:
    """
    Select the threshold that maximizes recall among those meeting a minimum
    precision bar, relative to the 0.5 baseline. Returns the chosen row plus
    the baseline row for comparison.
    """
    baseline = min(sweep_results, key=lambda r: abs(r["threshold"] - baseline_threshold))
    candidates = [r for r in sweep_results if r["precision"] >= min_precision]
    if not candidates:
        candidates = sweep_results
    best = max(candidates, key=lambda r: r["recall"])

    return {
        "baseline": baseline,
        "chosen": best,
        "recall_gain_pct": round((best["recall"] - baseline["recall"]) * 100, 2),
        "precision_change_pct": round((best["precision"] - baseline["precision"]) * 100, 2),
    }
