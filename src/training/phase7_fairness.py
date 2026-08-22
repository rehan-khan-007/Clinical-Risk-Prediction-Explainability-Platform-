"""
Phase 7: Fairness audit — check whether the model's performance (recall,
precision, specificity) differs meaningfully across age and sex subgroups
on the held-out test set. A model that looks good in aggregate can still
under-serve specific groups; this makes that visible rather than assumed.

Run:
    python -m src.training.phase7_fairness --config configs/cardio.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import evaluate_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict):
    d = Path(cfg["data"]["processed_dir"])
    X_train = pd.read_csv(d / "cardio_X_train.csv")
    y_train = pd.read_csv(d / "cardio_y_train.csv").squeeze()
    X_test = pd.read_csv(d / "cardio_X_test.csv")
    y_test = pd.read_csv(d / "cardio_y_test.csv").squeeze()
    X_test_unscaled = pd.read_csv(d / "cardio_X_test_unscaled.csv")
    return X_train, y_train, X_test, y_test, X_test_unscaled


def train_model(X_train, y_train, cfg: dict):
    m = cfg["models"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=m["n_estimators"], max_depth=m["max_depth"],
        random_state=m["random_state"], n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def make_age_bins(age_years: pd.Series, n_bins: int = 6) -> pd.Series:
    """Equal-width age bins, labeled by range, e.g. '35-42'."""
    bins = pd.cut(age_years, bins=n_bins, precision=0)
    return bins.astype(str)


def audit_subgroups(model, X_test, y_test, X_test_unscaled, threshold: float, group_col: pd.Series, group_name: str):
    y_proba = model.predict_proba(X_test)[:, 1]
    results = []
    for group_value in sorted(group_col.unique(), key=str):
        mask = (group_col == group_value).values
        n = int(mask.sum())
        if n < 30:  # too few samples for a stable estimate
            continue
        metrics = evaluate_model(model, X_test[mask], y_test[mask], threshold=threshold)
        results.append({
            "group_dimension": group_name,
            "group_value": str(group_value),
            "n": n,
            **{k: v for k, v in metrics.items() if k != "threshold"},
        })
    return results


def plot_subgroup_recall(results: list, group_name: str, out_path: Path):
    rows = [r for r in results if r["group_dimension"] == group_name]
    if not rows:
        return
    labels = [r["group_value"] for r in rows]
    recalls = [r["recall"] for r in rows]
    ns = [r["n"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, recalls, color="#4a7fd6")
    overall_recall = np.average(recalls, weights=ns)
    ax.axhline(overall_recall, color="gray", linestyle="--", label=f"Overall recall ({overall_recall:.3f})")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall by {group_name} (n shown per bar)")
    ax.legend()
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n}",
                 ha="center", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cardio.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    X_train, y_train, X_test, y_test, X_test_unscaled = load_data(cfg)
    model = train_model(X_train, y_train, cfg)
    threshold = 0.45  # matches Phase 2's chosen operating threshold

    y_proba_test = model.predict_proba(X_test)[:, 1]
    overall = evaluate_model(model, X_test, y_test, threshold=threshold)
    print("Overall (whole test set):")
    for k, v in overall.items():
        if k != "threshold":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    all_results = []

    # Sex subgroups (1=female, 2=male per source dataset encoding)
    sex_labels = X_test_unscaled["gender"].map({1: "female", 2: "male"})
    sex_results = audit_subgroups(model, X_test, y_test, X_test_unscaled, threshold, sex_labels, "sex")
    all_results.extend(sex_results)
    print("\nBy sex:")
    for r in sex_results:
        print(f"  {r['group_value']:8s} (n={r['n']:5d}): recall={r['recall']:.4f}  "
              f"precision={r['precision']:.4f}  specificity={r['specificity']:.4f}")

    # Age subgroups (6 equal-width bins across the test set's age range)
    age_bins = make_age_bins(X_test_unscaled["age"], n_bins=6)
    age_results = audit_subgroups(model, X_test, y_test, X_test_unscaled, threshold, age_bins, "age")
    all_results.extend(age_results)
    print("\nBy age group:")
    for r in age_results:
        print(f"  {r['group_value']:12s} (n={r['n']:5d}): recall={r['recall']:.4f}  "
              f"precision={r['precision']:.4f}  specificity={r['specificity']:.4f}")

    # Flag largest disparities
    def recall_spread(rows):
        recalls = [r["recall"] for r in rows]
        return max(recalls) - min(recalls) if recalls else 0.0

    sex_spread = recall_spread(sex_results)
    age_spread = recall_spread(age_results)
    print(f"\nRecall spread across sex subgroups: {sex_spread:.4f}")
    print(f"Recall spread across age subgroups: {age_spread:.4f}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_subgroup_recall(all_results, "sex", out_dir / "phase7_recall_by_sex.png")
    plot_subgroup_recall(all_results, "age", out_dir / "phase7_recall_by_age.png")

    summary = {
        "overall": {k: v for k, v in overall.items() if k != "threshold"},
        "threshold": threshold,
        "subgroups": all_results,
        "recall_spread_sex": sex_spread,
        "recall_spread_age": age_spread,
    }
    with open(out_dir / "phase7_fairness_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_dir / 'phase7_fairness_audit.json'}")
    print(f"Saved plots -> {out_dir}/phase7_recall_by_sex.png, phase7_recall_by_age.png")


if __name__ == "__main__":
    main()
