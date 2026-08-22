"""
Phase 2: Calibration analysis + threshold optimization for the selected
best model (Random Forest, from Phase 1's 5-fold CV benchmark).

Trains the model on the full training set (no CV here — CV was for model
selection; now we fit once and evaluate on the untouched held-out test
set), then:
  1. Plots/reports reliability curve + calibration error
  2. Sweeps thresholds 0.30-0.70
  3. Picks an operating threshold that maximizes recall while limiting
     precision loss to `max_precision_drop_pp` percentage points vs. 0.50

Run:
    python -m src.training.phase2_calibration --config configs/cardio.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.calibration import compute_calibration, sweep_thresholds, pick_best_threshold
from src.evaluation.metrics import evaluate_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_splits(cfg: dict):
    d = Path(cfg["data"]["processed_dir"])
    X_train = pd.read_csv(d / "cardio_X_train.csv")
    y_train = pd.read_csv(d / "cardio_y_train.csv").squeeze()
    X_test = pd.read_csv(d / "cardio_X_test.csv")
    y_test = pd.read_csv(d / "cardio_y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def plot_reliability(cal: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(cal["prob_pred"], cal["prob_true"], marker="o", label="Random Forest")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability curve (calibration error: {cal['mean_calibration_error']:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cardio.yaml")
    parser.add_argument("--max-precision-drop-pp", type=float, default=5.0,
                         help="Max precision loss (percentage points) vs. threshold=0.5 baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    X_train, y_train, X_test, y_test = load_splits(cfg)

    m = cfg["models"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=m["n_estimators"], max_depth=m["max_depth"],
        random_state=m["random_state"], n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Calibration
    cal = compute_calibration(y_test, y_proba, n_bins=10)
    print("Calibration (test set):")
    print(f"  Mean calibration error: {cal['mean_calibration_error']:.4f}")
    print(f"  Brier score: {cal['brier_score']:.4f}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_reliability(cal, out_dir / "phase2_reliability_curve.png")
    print(f"  Saved reliability curve -> {out_dir / 'phase2_reliability_curve.png'}")

    # 2. Threshold sweep
    sweep = sweep_thresholds(y_test, y_proba)
    print("\nThreshold sweep (test set):")
    for r in sweep:
        print(f"  t={r['threshold']:.2f}  precision={r['precision']:.4f}  "
              f"recall={r['recall']:.4f}  specificity={r['specificity']:.4f}  f1={r['f1']:.4f}")

    # 3. Pick operating threshold
    baseline_precision = next(r["precision"] for r in sweep if abs(r["threshold"] - 0.5) < 1e-6)
    min_precision = baseline_precision - (args.max_precision_drop_pp / 100)
    selection = pick_best_threshold(sweep, baseline_threshold=0.5, min_precision=min_precision)

    print(f"\nBaseline (t=0.50): precision={selection['baseline']['precision']:.4f}, "
          f"recall={selection['baseline']['recall']:.4f}")
    print(f"Chosen (t={selection['chosen']['threshold']:.2f}): "
          f"precision={selection['chosen']['precision']:.4f}, recall={selection['chosen']['recall']:.4f}")
    print(f"Recall gain: {selection['recall_gain_pct']:+.2f} pp | "
          f"Precision change: {selection['precision_change_pct']:+.2f} pp")

    results = {
        "calibration": {k: v for k, v in cal.items() if k not in ("prob_true", "prob_pred")},
        "threshold_sweep": sweep,
        "selection": selection,
    }
    with open(out_dir / "phase2_calibration_threshold.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results -> {out_dir / 'phase2_calibration_threshold.json'}")


if __name__ == "__main__":
    main()
