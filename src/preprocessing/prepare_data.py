"""
Phase 1: Load raw data, split, preprocess, save processed splits.

Run:
    python -m src.preprocessing.prepare_data --config configs/baseline.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict) -> pd.DataFrame:
    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. "
            "Download the UCI/Cleveland heart disease dataset (1,025-record "
            "version) and place it there before running this script."
        )
    df = pd.read_csv(raw_path)

    n_before = len(df)
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    if n_dupes > 0:
        print(
            f"WARNING: raw dataset contains {n_dupes} exact duplicate rows "
            f"({n_before} -> {len(df)} after dedup). This is a known issue "
            "with the public '1,025-row' mirror of this dataset (it's the "
            "original 303-row Cleveland set with padded duplicates). "
            "Splitting before dedup causes train/test leakage and inflated "
            "metrics (e.g. 1.0 ROC-AUC) — deduping here fixes that."
        )
    return df


def split_data(df: pd.DataFrame, cfg: dict):
    target = cfg["data"]["target_column"]
    X, y = df.drop(columns=[target]), df[target]

    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]
    rs = cfg["data"]["random_state"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=rs
    )
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=rs
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, cfg: dict):
    if not cfg["preprocessing"]["scale_numeric"]:
        return X_train, X_val, X_test, None

    scaler_type = cfg["preprocessing"]["scaler"]
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    numeric_cols = X_train.select_dtypes(include="number").columns
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_val, X_test, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_raw(cfg)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test, cfg)

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
    ]:
        obj.to_csv(out_dir / f"{name}.csv", index=False)

    print(f"Saved processed splits to {out_dir}/")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


if __name__ == "__main__":
    main()
