"""
Phase 1 (cardio dataset): benchmark LR/RF/XGBoost with stratified k-fold CV.

Reports mean +/- std across folds rather than a single train/val split —
more defensible on a dataset this size and standard practice for model
selection before touching the held-out test set.

Run:
    python -m src.training.train_cardio --config configs/cardio.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_train_split(cfg: dict):
    d = Path(cfg["data"]["processed_dir"])
    X = pd.read_csv(d / "cardio_X_train.csv")
    y = pd.read_csv(d / "cardio_y_train.csv").squeeze()
    return X, y


def build_models(cfg: dict):
    m = cfg["models"]
    return {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=m["logistic_regression"]["max_iter"],
            class_weight=m["logistic_regression"]["class_weight"],
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=m["random_forest"]["n_estimators"],
            max_depth=m["random_forest"]["max_depth"],
            random_state=m["random_forest"]["random_state"],
            n_jobs=-1,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=m["xgboost"]["n_estimators"],
            max_depth=m["xgboost"]["max_depth"],
            learning_rate=m["xgboost"]["learning_rate"],
            eval_metric=m["xgboost"]["eval_metric"],
            random_state=m["xgboost"]["random_state"],
            n_jobs=-1,
        ),
    }


def run_cv(model_fn, X, y, cfg: dict) -> dict:
    cv_cfg = cfg["cross_validation"]
    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )

    fold_metrics = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        fold_metrics.append(evaluate_model(model, X_val, y_val, threshold=cfg["evaluation"]["threshold"]))

    agg = {}
    for key in fold_metrics[0]:
        if key == "threshold":
            continue
        vals = [f[key] for f in fold_metrics]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cardio.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    X, y = load_train_split(cfg)
    model_builders = build_models(cfg)

    n_splits = cfg["cross_validation"]["n_splits"]
    print(f"Running {n_splits}-fold stratified CV on {len(X)} training rows...\n")

    results = {}
    for name, model_fn in model_builders.items():
        agg = run_cv(model_fn, X, y, cfg)
        results[name] = agg
        print(f"{name}:")
        for metric, stats in agg.items():
            print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
        print()

    best_model = max(results, key=lambda k: results[k]["roc_auc"]["mean"])
    print(f"Best model by mean ROC-AUC across {n_splits} folds: {best_model} "
          f"({results[best_model]['roc_auc']['mean']:.4f} +/- {results[best_model]['roc_auc']['std']:.4f})")

    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "cardio_cv_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path / 'cardio_cv_benchmark.json'}")


if __name__ == "__main__":
    main()
