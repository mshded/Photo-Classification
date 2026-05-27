from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.classifier import train_and_save_model


def main() -> None:
    results = train_and_save_model(
        labels_csv_path="data/labels.csv",
        model_path="models/best_model.pkl",
        model_type="logreg",
    )

    metrics_rows = []
    for split_name in ("train", "val", "test"):
        m = results[f"{split_name}_metrics"].copy()
        m["split"] = split_name
        m["threshold"] = results["threshold"]
        m["predicted_positive"] = int(m.get("tp", 0) + m.get("fp", 0))
        metrics_rows.append(m)

    metrics = pd.DataFrame(metrics_rows)[[
        "split", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn", "n", "predicted_positive", "threshold"
    ]]

    Path("results").mkdir(parents=True, exist_ok=True)
    metrics.to_csv("results/metrics.csv", index=False)
    results["threshold_table"].to_csv("results/threshold_metrics.csv", index=False)
    results["split_assignment"].to_csv("results/split_assignment.csv", index=False)

    print(metrics.to_string(index=False))
    print("Saved: models/best_model.pkl, results/metrics.csv, results/threshold_metrics.csv, results/split_assignment.csv")


if __name__ == "__main__":
    main()
