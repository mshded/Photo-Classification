from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_stage8_experiments


if __name__ == "__main__":
    metrics = run_stage8_experiments(
        labels_csv_path="data/labels.csv",
        metrics_output_path="results/metrics.csv",
        model_output_path="models/best_model.pkl",
    )
    print(metrics.to_string(index=False))
