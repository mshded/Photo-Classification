from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.classifier import (
    build_model_pipeline,
    load_labeled_data,
    predict_proba,
    save_model_artifacts,
)
from src.features import build_ml_feature_frame
from src.metrics import compute_classification_metrics, select_threshold_for_precision
from src.pipeline import apply_baseline_rules


LABEL_MAP = {"content": 1, "non_content": 0}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "t"})


def prepare_labeled_data(labels_csv_path: str = "data/labels.csv") -> pd.DataFrame:
    df = load_labeled_data(labels_csv_path=labels_csv_path).copy()
    df["target"] = df["label"].map(LABEL_MAP).astype(int)

    if "is_downloaded_and_valid" in df.columns:
        is_valid = _as_bool(df["is_downloaded_and_valid"])
    else:
        is_valid = pd.Series([True] * len(df), index=df.index)

    df["download_ok"] = is_valid
    df["is_valid_image"] = is_valid

    return df


def evaluate_heuristics_only(df: pd.DataFrame) -> dict[str, Any]:
    test_df = df[df["split"] == "test"].copy()
    baseline_df = apply_baseline_rules(test_df)
    y_true = baseline_df["target"].astype(int)
    y_pred = baseline_df["baseline_keep"].fillna(False).astype(int)

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred)
    metrics["threshold"] = pd.NA
    metrics["notes"] = "Stage 6 baseline rules on test split"
    return metrics


def _fit_ml(df: pd.DataFrame) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    features = build_ml_feature_frame(df)

    model = build_model_pipeline(model_type="logreg")
    model.fit(features.loc[train_df.index], train_df["target"])

    train_proba = pd.Series(model.predict_proba(features.loc[train_df.index])[:, 1], index=train_df.index)
    val_proba = pd.Series(model.predict_proba(features.loc[val_df.index])[:, 1], index=val_df.index)
    test_proba = pd.Series(model.predict_proba(features.loc[test_df.index])[:, 1], index=test_df.index)
    return model, train_df, val_df, test_df, train_proba, val_proba, test_proba


def evaluate_ml_only(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    model, train_df, val_df, test_df, _, val_proba, test_proba = _fit_ml(df)

    threshold, threshold_table = select_threshold_for_precision(
        y_true=val_df["target"],
        y_proba=val_proba,
        tie_breaker="recall",
    )

    test_pred = (test_proba >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true=test_df["target"], y_pred=test_pred)
    metrics["threshold"] = float(threshold)
    metrics["notes"] = "LogReg, threshold tuned on val for max precision"

    artifacts = {
        "model": model,
        "threshold": float(threshold),
        "model_type": "logreg",
        "numeric_features": [
            "width",
            "height",
            "area",
            "aspect_ratio",
            "file_size_bytes",
            "is_tiny",
            "is_suspicious_domain",
            "has_ui_keyword",
            "is_too_small",
            "has_extreme_aspect_ratio",
            "has_tracking_hint",
            "has_suspicious_keyword",
        ],
        "categorical_features": ["format"],
    }
    return metrics, {"artifacts": artifacts, "threshold_table": threshold_table}


def evaluate_heuristics_plus_ml(df: pd.DataFrame, model: Any) -> tuple[dict[str, Any], float, pd.DataFrame]:
    train_df = apply_baseline_rules(df[df["split"] == "train"].copy())
    val_df = apply_baseline_rules(df[df["split"] == "val"].copy())
    test_df = apply_baseline_rules(df[df["split"] == "test"].copy())

    val_mask = val_df["baseline_keep"].fillna(False)
    test_mask = test_df["baseline_keep"].fillna(False)

    val_threshold_table = pd.DataFrame()
    if val_mask.any():
        val_scores = predict_proba(model, val_df.loc[val_mask])
        threshold, val_threshold_table = select_threshold_for_precision(
            y_true=val_df.loc[val_mask, "target"],
            y_proba=val_scores,
            tie_breaker="recall",
        )
    else:
        threshold = 0.5

    test_pred = pd.Series(0, index=test_df.index, dtype=int)
    if test_mask.any():
        test_scores = predict_proba(model, test_df.loc[test_mask])
        test_pred.loc[test_mask] = (test_scores >= threshold).astype(int)

    metrics = compute_classification_metrics(y_true=test_df["target"], y_pred=test_pred)
    metrics["threshold"] = float(threshold)
    metrics["notes"] = "Stage 6 rules prefilter + LogReg for remaining candidates"
    return metrics, float(threshold), val_threshold_table


def _build_result_row(
    *,
    experiment_name: str,
    method_type: str,
    uses_heuristics: bool,
    uses_ml: bool,
    model_name: str,
    feature_type: str,
    threshold: Any,
    metrics: dict[str, Any],
    n_train: int,
    n_val: int,
    n_test: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "method_type": method_type,
        "uses_heuristics": uses_heuristics,
        "uses_ml": uses_ml,
        "model_name": model_name,
        "feature_type": feature_type,
        "threshold": threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "notes": notes,
    }


def run_stage8_experiments(
    labels_csv_path: str = "data/labels.csv",
    metrics_output_path: str = "results/metrics.csv",
    model_output_path: str = "models/best_model.pkl",
) -> pd.DataFrame:
    df = prepare_labeled_data(labels_csv_path=labels_csv_path)
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_test = int((df["split"] == "test").sum())

    heur_metrics = evaluate_heuristics_only(df)

    ml_metrics, ml_meta = evaluate_ml_only(df)
    save_model_artifacts(ml_meta["artifacts"], model_path=model_output_path)

    combo_metrics, combo_threshold, _ = evaluate_heuristics_plus_ml(df, model=ml_meta["artifacts"]["model"])

    rows = [
        _build_result_row(
            experiment_name="heuristics_only_default",
            method_type="heuristic_only",
            uses_heuristics=True,
            uses_ml=False,
            model_name="rules_v1",
            feature_type="metadata+url_flags",
            threshold=heur_metrics["threshold"],
            metrics=heur_metrics,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            notes=heur_metrics["notes"],
        ),
        _build_result_row(
            experiment_name="ml_only_threshold_tuned",
            method_type="ml_only",
            uses_heuristics=False,
            uses_ml=True,
            model_name="logreg",
            feature_type="tabular_metadata+heuristic_flags",
            threshold=ml_metrics["threshold"],
            metrics=ml_metrics,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            notes=ml_metrics["notes"],
        ),
        _build_result_row(
            experiment_name="heuristics_plus_ml_threshold_tuned",
            method_type="heuristic_plus_ml",
            uses_heuristics=True,
            uses_ml=True,
            model_name="rules_v1+logreg",
            feature_type="rules_prefilter+tabular_metadata",
            threshold=combo_threshold,
            metrics=combo_metrics,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            notes=combo_metrics["notes"],
        ),
    ]

    result_df = pd.DataFrame(rows)
    out_path = Path(metrics_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    return result_df


if __name__ == "__main__":
    df = run_stage8_experiments()
    print(df.to_string(index=False))
