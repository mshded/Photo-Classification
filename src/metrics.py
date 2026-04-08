from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = pd.Series(list(y_true)).astype(int)
    y_pred = pd.Series(list(y_pred)).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": int(len(y_true)),
    }


def evaluate_baseline_on_labels(labels_csv_path: str) -> Dict[str, float] | pd.DataFrame:
    labels_path = Path(labels_csv_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv_path}")

    df = pd.read_csv(labels_path)
    if "label" not in df.columns:
        raise ValueError("labels.csv must contain 'label' column")

    if "baseline_keep" not in df.columns:
        # Lightweight fallback: evaluate proxy heuristic columns if baseline output is not merged yet.
        if {"is_tiny", "is_suspicious_domain", "has_ui_keyword"}.issubset(df.columns):
            proxy_reject = (
                df["is_tiny"].fillna(False)
                | df["is_suspicious_domain"].fillna(False)
                | df["has_ui_keyword"].fillna(False)
            )
            y_pred = (~proxy_reject).astype(int)
        else:
            return pd.DataFrame(
                [
                    {
                        "status": "baseline_keep column is missing",
                        "hint": "merge baseline_results.csv with labels.csv by candidate_id/image_url/local_path",
                    }
                ]
            )
    else:
        y_pred = df["baseline_keep"].fillna(False).astype(int)

    y_true = df["label"].map({"content": 1, "non_content": 0}).fillna(0).astype(int)
    return compute_classification_metrics(y_true=y_true, y_pred=y_pred)
