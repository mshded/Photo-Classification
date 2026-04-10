from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import build_ml_feature_frame
from src.metrics import evaluate_model_on_split, select_threshold_for_precision

LABEL_MAP = {"content": 1, "non_content": 0}


NUMERIC_FEATURES = [
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
]
CATEGORICAL_FEATURES = ["format"]


def load_labeled_data(labels_csv_path: str = "data/labels.csv") -> pd.DataFrame:
    df = pd.read_csv(labels_csv_path)
    out = df.copy()
    out["target"] = out["label"].map(LABEL_MAP)
    out = out[out["target"].isin([0, 1])].copy()

    if "split" in out.columns and out["split"].fillna("").str.strip().ne("").any():
        out["split"] = out["split"].fillna("").astype(str).str.strip().str.lower()
        return out

    train_val_idx, test_idx = train_test_split(
        out.index,
        test_size=0.2,
        random_state=42,
        stratify=out["target"],
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=42,
        stratify=out.loc[train_val_idx, "target"],
    )

    out["split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def build_model_pipeline(model_type: str = "logreg") -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def predict_proba(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    feature_df = build_ml_feature_frame(df)
    return pd.Series(model.predict_proba(feature_df)[:, 1], index=df.index, name="ml_score")


def save_model_artifacts(artifacts: dict[str, Any], model_path: str = "models/best_model.pkl") -> None:
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, target)


def load_model_artifacts(model_path: str = "models/best_model.pkl") -> dict[str, Any]:
    target = Path(model_path)
    return joblib.load(target)


def train_and_save_model(
    labels_csv_path: str = "data/labels.csv",
    model_path: str = "models/best_model.pkl",
    model_type: str = "logreg",
    ) -> dict[str, Any]:
    df = load_labeled_data(labels_csv_path=labels_csv_path)
    features = build_ml_feature_frame(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    model = build_model_pipeline(model_type=model_type)
    model.fit(features.loc[train_df.index], train_df["target"])

    val_proba = pd.Series(model.predict_proba(features.loc[val_df.index])[:, 1], index=val_df.index)
    threshold, threshold_table = select_threshold_for_precision(val_df["target"], val_proba)

    train_pred = (pd.Series(model.predict_proba(features.loc[train_df.index])[:, 1], index=train_df.index) >= threshold).astype(int)
    val_pred = (val_proba >= threshold).astype(int)
    test_proba = pd.Series(model.predict_proba(features.loc[test_df.index])[:, 1], index=test_df.index)
    test_pred = (test_proba >= threshold).astype(int)

    artifacts = {
        "model": model,
        "threshold": float(threshold),
        "model_type": model_type,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    save_model_artifacts(artifacts=artifacts, model_path=model_path)

    return {
        "threshold": float(threshold),
        "train_metrics": evaluate_model_on_split(train_df["target"], train_pred),
        "val_metrics": evaluate_model_on_split(val_df["target"], val_pred, val_proba),
        "test_metrics": evaluate_model_on_split(test_df["target"], test_pred, test_proba),
        "threshold_table": threshold_table,
    }