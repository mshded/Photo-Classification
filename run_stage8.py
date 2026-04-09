from __future__ import annotations

from src.experiments import run_stage8_experiments


if __name__ == "__main__":
    metrics = run_stage8_experiments(
        labels_csv_path="data/labels.csv",
        metrics_output_path="results/metrics.csv",
        model_output_path="models/best_model.pkl",
    )
    print(metrics.to_string(index=False))
