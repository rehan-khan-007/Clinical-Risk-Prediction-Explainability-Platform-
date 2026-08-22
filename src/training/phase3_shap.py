"""
Phase 3: SHAP-based explainability for the selected model (Random Forest).

Produces:
  1. Global explanation — SHAP summary plot + mean |SHAP value| per feature
     across the held-out test set (what matters overall)
  2. Local explanations — waterfall plots for a handful of individual
     high-risk predictions (why *this* patient specifically)

Run:
    python -m src.training.phase3_shap --config configs/cardio.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.ensemble import RandomForestClassifier


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


def load_unscaled_test(cfg: dict) -> pd.DataFrame:
    """
    Row-aligned, original-unit version of the test set (mmHg, kg, years),
    saved by prepare_cardio.py. SHAP values are computed on the scaled
    features (matches the trained model), but displaying standardized
    values like 'ap_hi = 4.403' in a waterfall plot is meaningless to a
    reader — this gives us real units to display instead.
    """
    d = Path(cfg["data"]["processed_dir"])
    return pd.read_csv(d / "cardio_X_test_unscaled.csv")


def train_model(X_train, y_train, cfg: dict):
    m = cfg["models"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=m["n_estimators"], max_depth=m["max_depth"],
        random_state=m["random_state"], n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def global_explanation(model, X_sample, out_dir: Path, top_k: int = 5):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary RandomForestClassifier, shap_values is [class0, class1] or
    # a single 3D array depending on SHAP version — normalize to class-1 array
    if isinstance(shap_values, list):
        sv_pos = shap_values[1]
    elif shap_values.ndim == 3:
        sv_pos = shap_values[:, :, 1]
    else:
        sv_pos = shap_values

    mean_abs = np.abs(sv_pos).mean(axis=0)
    importance = sorted(
        zip(X_sample.columns, mean_abs), key=lambda x: x[1], reverse=True
    )

    print(f"Global feature importance (mean |SHAP value|, top {top_k}):")
    for feat, val in importance[:top_k]:
        print(f"  {feat}: {val:.4f}")

    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(sv_pos, X_sample, show=False)
    plt.tight_layout()
    fig.savefig(out_dir / "phase3_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved global summary plot -> {out_dir / 'phase3_shap_summary.png'}")

    return explainer, sv_pos, [{"feature": f, "mean_abs_shap": float(v)} for f, v in importance]


def local_explanations(model, explainer, X_sample, X_sample_unscaled, y_proba, out_dir: Path, n_examples: int = 3, top_k: int = 5):
    # Pick the n_examples highest-confidence positive predictions
    top_idx = np.argsort(y_proba)[-n_examples:][::-1]

    local_results = []
    for rank, idx in enumerate(top_idx, start=1):
        row = X_sample.iloc[[idx]]
        row_unscaled = X_sample_unscaled.iloc[[idx]]
        shap_values = explainer.shap_values(row)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base_value = explainer.expected_value[1]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
            base_value = explainer.expected_value[1]
        else:
            sv = shap_values[0]
            base_value = explainer.expected_value

        contributions = sorted(
            zip(X_sample.columns, sv), key=lambda x: abs(x[1]), reverse=True
        )[:top_k]

        print(f"\nPatient #{idx} (risk probability: {y_proba[idx]:.4f}):")
        for feat, val in contributions:
            direction = "increases" if val > 0 else "decreases"
            print(f"  {feat}: {val:+.4f} ({direction} risk)")

        local_results.append({
            "row_index": int(idx),
            "risk_probability": float(y_proba[idx]),
            "top_contributions": [
                {"feature": f, "shap_value": float(v)} for f, v in contributions
            ],
        })

        exp = shap.Explanation(
            values=sv, base_values=base_value,
            data=row_unscaled.values[0], feature_names=list(X_sample.columns),
        )
        fig = plt.figure(figsize=(7, 4))
        shap.plots.waterfall(exp, show=False, max_display=top_k + 2)
        plt.tight_layout()
        fig.savefig(out_dir / f"phase3_waterfall_patient{rank}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved {n_examples} local waterfall plots -> {out_dir}/phase3_waterfall_patient*.png")
    return local_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cardio.yaml")
    parser.add_argument("--sample-size", type=int, default=1000,
                         help="Subsample of test set for global SHAP (full test set is slow with TreeExplainer)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    X_train, y_train, X_test, y_test = load_splits(cfg)
    model = train_model(X_train, y_train, cfg)

    X_test_unscaled = load_unscaled_test(cfg)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(args.sample_size, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    X_sample_unscaled = X_test_unscaled.iloc[sample_idx].reset_index(drop=True)
    y_proba_sample = model.predict_proba(X_sample)[:, 1]

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    explainer, sv_pos, global_importance = global_explanation(model, X_sample, out_dir, top_k=args.top_k)
    local_results = local_explanations(model, explainer, X_sample, X_sample_unscaled, y_proba_sample, out_dir, n_examples=3, top_k=args.top_k)

    results = {
        "sample_size": len(X_sample),
        "global_importance": global_importance,
        "local_examples": local_results,
    }
    with open(out_dir / "phase3_shap.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results -> {out_dir / 'phase3_shap.json'}")


if __name__ == "__main__":
    main()
