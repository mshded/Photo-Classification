from __future__ import annotations

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
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": int(len(y_true)),
    }


def evaluate_model_on_split(
    y_true: Iterable[int], y_pred: Iterable[int], y_proba: Iterable[float] | None = None
    ) -> Dict[str, float]:
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred)
    if y_proba is not None:
        proba_series = pd.Series(list(y_proba), dtype=float)
        metrics.update(
            {
                "mean_proba": float(proba_series.mean()),
                "median_proba": float(proba_series.median()),
                "proba_p90": float(proba_series.quantile(0.9)),
            }
        )
    return metrics


def build_threshold_metrics_table(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    y_true_s = pd.Series(list(y_true)).astype(int)
    y_proba_s = pd.Series(list(y_proba)).astype(float)

    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 100, 5)]

    rows = []
    for t in thresholds:
        y_pred = (y_proba_s >= float(t)).astype(int)
        row = compute_classification_metrics(y_true=y_true_s, y_pred=y_pred)
        row["threshold"] = float(t)
        rows.append(row)

    return pd.DataFrame(rows)


def select_threshold_for_precision(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    min_positive_predictions: int = 1,
    min_precision: float = 0.90,
    tie_breaker: str = "recall",
    ) -> tuple[float, pd.DataFrame]:
    metrics_table = build_threshold_metrics_table(y_true=y_true, y_proba=y_proba)

    tie_breaker = tie_breaker if tie_breaker in {"recall", "f1"} else "recall"

    valid = metrics_table[metrics_table["tp"] + metrics_table["fp"] >= min_positive_predictions].copy()
    if valid.empty:
        valid = metrics_table.copy()

    precision_filtered = valid[valid["precision"] >= min_precision].copy()

    if not precision_filtered.empty:
        sorted_table = precision_filtered.sort_values(
            by=[tie_breaker, "f1", "precision", "threshold"],
            ascending=[False, False, False, False],
        )
    else:
        sorted_table = valid.sort_values(
            by=["precision", tie_breaker, "f1", "threshold"],
            ascending=[False, False, False, False],
        )

    best_row = sorted_table.iloc[0]
    return float(best_row["threshold"]), metrics_table.sort_values("threshold").reset_index(drop=True)
