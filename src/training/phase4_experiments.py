"""
Phase 4: Run config-driven preprocessing/model variants, logging every run
to MLflow (params, metrics, model artifact, plots) so any past result is
reproducible from its logged config alone.

Each variant re-runs preprocessing with its own scaler choice (since scaling
is fit on train and must match what the model was trained on), then trains
the specified model and evaluates on the same held-out test set used in
Phases 1-3, so results are directly comparable across variants.

Run:
    python -m src.training.phase4_experiments --experiments configs/experiments.yaml
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_model
from src.preprocessing.prepare_cardio import load_raw, clean_data


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Short hash of the effective config, for tracking exact reproducibility."""
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def build_variant_config(base_cfg: dict, variant: dict) -> dict:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy
    if "preprocessing" in variant:
        cfg["preprocessing"].update(variant["preprocessing"])
    return cfg


def prepare_variant_data(cfg: dict):
    """Re-run cleaning + split + scaling for this variant's scaler choice."""
    from sklearn.model_selection import train_test_split

    df = load_raw(cfg)
    df = clean_data(df, cfg)

    target = cfg["data"]["target_column"]
    X, y = df.drop(columns=[target]), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"],
        stratify=y, random_state=cfg["data"]["random_state"],
    )

    scaler_type = cfg["preprocessing"]["scaler"]
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    numeric_cols = X_train.select_dtypes(include="number").columns
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test


def build_model(model_name: str, cfg: dict, overrides: dict):
    m = dict(cfg["models"][model_name])
    m.update(overrides or {})

    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=m["max_iter"], class_weight=m["class_weight"])
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=m["n_estimators"], max_depth=m["max_depth"],
            random_state=m["random_state"], n_jobs=-1,
        )
    elif model_name == "xgboost":
        return XGBClassifier(
            n_estimators=m["n_estimators"], max_depth=m["max_depth"],
            learning_rate=m["learning_rate"], eval_metric=m["eval_metric"],
            random_state=m["random_state"], n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="clinical-risk-platform")
    args = parser.parse_args()

    exp_cfg = load_yaml(args.experiments)
    base_cfg = load_yaml(exp_cfg["base_config"])

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    summary = []

    for variant in exp_cfg["variants"]:
        name = variant["name"]
        model_name = variant["model"]
        overrides = variant.get("model_overrides", {})
        cfg = build_variant_config(base_cfg, variant)
        run_config_hash = config_hash({"variant": variant, "base": base_cfg})

        print(f"\n=== Run: {name} ===")
        print(f"  {variant.get('description', '')}")

        X_train, X_test, y_train, y_test = prepare_variant_data(cfg)
        model = build_model(model_name, cfg, overrides)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, threshold=cfg["evaluation"]["threshold"])

        with mlflow.start_run(run_name=name):
            mlflow.log_param("config_hash", run_config_hash)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("scaler", cfg["preprocessing"]["scaler"])
            for k, v in overrides.items():
                mlflow.log_param(f"override_{k}", v)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

            for metric_name, value in metrics.items():
                if metric_name != "threshold":
                    mlflow.log_metric(metric_name, value)

            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = mlflow.active_run().info.run_id

        print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | run_id: {run_id}")
        summary.append({
            "name": name, "model": model_name, "config_hash": run_config_hash,
            "run_id": run_id, **{k: v for k, v in metrics.items() if k != "threshold"},
        })

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "phase4_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best = max(summary, key=lambda r: r["roc_auc"])
    print(f"\n=== Summary ({len(summary)} runs) ===")
    for r in sorted(summary, key=lambda r: -r["roc_auc"]):
        print(f"  {r['name']:35s} roc_auc={r['roc_auc']:.4f}  f1={r['f1']:.4f}  run_id={r['run_id'][:8]}")
    print(f"\nBest: {best['name']} (ROC-AUC {best['roc_auc']:.4f}, config_hash={best['config_hash']})")
    print(f"Saved summary -> {out_dir / 'phase4_experiment_summary.json'}")
    print(f"MLflow UI: run `mlflow ui --backend-store-uri {args.tracking_uri}` and open http://localhost:5000")


if __name__ == "__main__":
    main()
