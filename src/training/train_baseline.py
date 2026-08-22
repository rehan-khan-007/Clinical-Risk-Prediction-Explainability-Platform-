"""
Phase 1: Train LR, RF, XGBoost on processed splits and benchmark them.

Run:
    python -m src.training.train_baseline --config configs/baseline.yaml
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_splits(cfg: dict):
    d = Path(cfg["data"]["processed_dir"])
    X_train = pd.read_csv(d / "X_train.csv")
    X_val = pd.read_csv(d / "X_val.csv")
    y_train = pd.read_csv(d / "y_train.csv").squeeze()
    y_val = pd.read_csv(d / "y_val.csv").squeeze()
    return X_train, X_val, y_train, y_val


def build_models(cfg: dict):
    m = cfg["models"]
    return {
        "logistic_regression": LogisticRegression(
            max_iter=m["logistic_regression"]["max_iter"],
            class_weight=m["logistic_regression"]["class_weight"],
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=m["random_forest"]["n_estimators"],
            max_depth=m["random_forest"]["max_depth"],
            random_state=m["random_forest"]["random_state"],
        ),
        "xgboost": XGBClassifier(
            n_estimators=m["xgboost"]["n_estimators"],
            max_depth=m["xgboost"]["max_depth"],
            learning_rate=m["xgboost"]["learning_rate"],
            eval_metric=m["xgboost"]["eval_metric"],
            random_state=m["xgboost"]["random_state"],
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    X_train, X_val, y_train, y_val = load_splits(cfg)
    models = build_models(cfg)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val, threshold=cfg["evaluation"]["threshold"])
        results[name] = metrics
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    best_model = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_model} ({results[best_model]['roc_auc']:.4f})")

    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "baseline_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_path / 'baseline_benchmark.json'}")


if __name__ == "__main__":
    main()
