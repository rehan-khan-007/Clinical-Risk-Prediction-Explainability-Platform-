"""
Phase 1 (cardio dataset): load, clean, and split the 70k-record
cardiovascular disease dataset.

This dataset is real patient data with real data-entry problems (negative
blood pressure readings, systolic < diastolic, age stored in days). This
script documents and handles each one explicitly rather than silently
dropping rows, so every cleaning decision is traceable and defensible.

Run:
    python -m src.preprocessing.prepare_cardio --config configs/cardio.yaml
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
        raise FileNotFoundError(f"Dataset not found at {raw_path}.")
    df = pd.read_csv(raw_path, sep=cfg["data"]["delimiter"])

    id_col = cfg["data"].get("id_column")
    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])

    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped {n_dupes} exact duplicate rows.")
    return df


def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["cleaning"]
    n_start = len(df)
    report = {}

    if c.get("age_in_days"):
        df = df.copy()
        df["age"] = (df["age"] / 365.25).round(1)

    lo, hi = c["ap_hi_range"]
    mask = df["ap_hi"].between(lo, hi)
    report["ap_hi_out_of_range"] = (~mask).sum()
    df = df[mask]

    lo, hi = c["ap_lo_range"]
    mask = df["ap_lo"].between(lo, hi)
    report["ap_lo_out_of_range"] = (~mask).sum()
    df = df[mask]

    if c.get("require_hi_gte_lo"):
        mask = df["ap_hi"] >= df["ap_lo"]
        report["systolic_lt_diastolic"] = (~mask).sum()
        df = df[mask]

    lo, hi = c["height_range"]
    mask = df["height"].between(lo, hi)
    report["height_out_of_range"] = (~mask).sum()
    df = df[mask]

    lo, hi = c["weight_range"]
    mask = df["weight"].between(lo, hi)
    report["weight_out_of_range"] = (~mask).sum()
    df = df[mask]

    df = df.reset_index(drop=True)
    n_end = len(df)

    print("Cleaning report:")
    for k, v in report.items():
        print(f"  {k}: {v} rows flagged")
    print(f"  Total: {n_start} -> {n_end} rows ({n_start - n_end} dropped, "
          f"{(n_start - n_end) / n_start * 100:.1f}%)")

    return df


def split_and_scale(df: pd.DataFrame, cfg: dict):
    target = cfg["data"]["target_column"]
    X, y = df.drop(columns=[target]), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        stratify=y,
        random_state=cfg["data"]["random_state"],
    )

    X_test_unscaled = X_test.copy()

    if cfg["preprocessing"]["scale_numeric"]:
        scaler_type = cfg["preprocessing"]["scaler"]
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        numeric_cols = X_train.select_dtypes(include="number").columns
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, X_test_unscaled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cardio.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_raw(cfg)
    df = clean_data(df, cfg)
    X_train, X_test, y_train, y_test, X_test_unscaled = split_and_scale(df, cfg)

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in [
        ("cardio_X_train", X_train), ("cardio_X_test", X_test),
        ("cardio_y_train", y_train), ("cardio_y_test", y_test),
        ("cardio_X_test_unscaled", X_test_unscaled),
    ]:
        obj.to_csv(out_dir / f"{name}.csv", index=False)

    print(f"\nSaved to {out_dir}/ — Train: {len(X_train)} | Held-out test: {len(X_test)}")
    print("(Train set will be used for k-fold CV in Phase 1 training; "
          "test set stays untouched until final evaluation.)")


if __name__ == "__main__":
    main()
