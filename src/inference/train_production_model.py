"""
Phase 5: Train the selected model (XGBoost — winner of Phase 4's variant
sweep at ROC-AUC 0.7996) on the full training set and persist it as an
artifact the API loads at startup, rather than retraining per request.

Also saves the fitted scaler, since incoming API requests arrive in raw
units (mmHg, kg, years) and must be scaled identically to training data.

Run:
    python -m src.inference.train_production_model --config configs/cardio.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_model
from src.preprocessing.prepare_cardio import load_raw, clean_data


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cardio.yaml")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_raw(cfg)
    df = clean_data(df, cfg)

    from sklearn.model_selection import train_test_split
    target = cfg["data"]["target_column"]
    X, y = df.drop(columns=[target]), df[target]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"],
        stratify=y, random_state=cfg["data"]["random_state"],
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    numeric_cols = X_train.select_dtypes(include="number").columns
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    m = cfg["models"]["xgboost"]
    model = XGBClassifier(
        n_estimators=m["n_estimators"], max_depth=m["max_depth"],
        learning_rate=m["learning_rate"], eval_metric=m["eval_metric"],
        random_state=m["random_state"], n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    metrics = evaluate_model(model, X_test_scaled, y_test, threshold=0.45)
    print("Production model test-set performance (threshold=0.45):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    version = "v1.0.0-xgboost"
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    metadata = {
        "version": version,
        "model_type": "xgboost",
        "feature_columns": feature_columns,
        "threshold": 0.45,
        "test_metrics": {k: v for k, v in metrics.items() if k != "threshold"},
        "trained_on_rows": len(X_train),
    }
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model artifact -> {out_dir}/model.joblib")
    print(f"Saved scaler -> {out_dir}/scaler.joblib")
    print(f"Saved metadata -> {out_dir}/model_metadata.json")
    print(f"Model version: {version}")


if __name__ == "__main__":
    main()
