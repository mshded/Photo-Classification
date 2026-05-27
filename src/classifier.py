from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
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
    "has_hard_block_keyword",
    "repeated_url_count",
    "alt_text_length",
    "file_name_length",
    "url_depth",
    "is_large_image",
    "has_descriptive_alt",
]
CATEGORICAL_FEATURES = ["format", "source_attr"]

THRESHOLD_MIN_PRECISION = 0.80
THRESHOLD_MIN_POSITIVE_PREDICTIONS = 3
THRESHOLD_TIE_BREAKER = "f1"

SIZE_QUERY_KEYS = {
    "w", "h", "width", "height", "size", "sz", "dpr", "quality", "q", "crop", "fit", "resize"
}
RESIZE_TOKEN_RE = re.compile(r"(?<![a-z0-9])\d{2,4}x\d{2,4}(?![a-z0-9])")
EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif|bmp|tiff?)$", flags=re.IGNORECASE)


def _normalize_image_url(image_url: str) -> str:
    if not image_url:
        return ""
    parsed = urlsplit(str(image_url).strip())
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, query, ""))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_local_path(local_path: str | Path | None) -> str:
    """Normalize saved local paths to a portable project-relative form."""
    if not local_path or pd.isna(local_path):
        return ""

    text = str(local_path).strip().replace("\\", "/")
    marker_idx = text.lower().find("/data/")
    if marker_idx >= 0:
        text = text[marker_idx + 1 :]

    cleaned = Path(text)
    try:
        if cleaned.is_absolute():
            return cleaned.resolve().relative_to(_project_root().resolve()).as_posix()
    except Exception:
        return cleaned.as_posix()
    return cleaned.as_posix()


def _canonicalize_for_grouping(image_url: str) -> str:
    """Map URL render/resize variants to the same duplicate-safe identifier."""
    if not image_url:
        return ""

    parsed = urlsplit(str(image_url).strip())
    path = RESIZE_TOKEN_RE.sub("", parsed.path or "")
    path = re.sub(r"/{2,}", "/", path)
    path = EXT_RE.sub("", path).rstrip("/")

    query_pairs = [
        (key.lower(), value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in SIZE_QUERY_KEYS
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path.lower(), query, ""))


def build_group_id(df: pd.DataFrame) -> pd.Series:
    """Build reproducible duplicate-safe groups only from stored CSV fields."""
    image_url = (
        df.get("image_url", pd.Series("", index=df.index, dtype="object"))
        .fillna("")
        .astype(str)
    )
    canonical_url = image_url.apply(_canonicalize_for_grouping)
    normalized_url = image_url.apply(_normalize_image_url)
    candidate_id = (
        df.get("candidate_id", pd.Series("", index=df.index, dtype="object"))
        .fillna("")
        .astype(str)
    )

    group_id = canonical_url.copy()
    group_id = group_id.where(group_id.str.len() > 0, normalized_url)
    group_id = group_id.where(group_id.str.len() > 0, candidate_id)
    group_id = group_id.where(group_id.str.len() > 0, df.index.astype(str))
    return group_id.astype(str)


def build_page_group_id(df: pd.DataFrame) -> pd.Series:
    """Return page identifiers for optional page-holdout diagnostics."""
    if "page_stub" in df.columns and df["page_stub"].fillna("").astype(str).str.len().gt(0).any():
        return df["page_stub"].fillna("").astype(str)
    if "page_id" in df.columns and df["page_id"].fillna("").astype(str).str.len().gt(0).any():
        return df["page_id"].fillna("").astype(str)
    if "page_url" in df.columns and df["page_url"].fillna("").astype(str).str.len().gt(0).any():
        return df["page_url"].fillna("").astype(str)
    return build_group_id(df)


def _assign_group_splits(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Create deterministic train/validation/test splits without duplicate leakage."""
    out = df.copy()
    groups = build_group_id(out)

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_idx, test_idx = next(test_splitter.split(out, out["target"], groups=groups))

    train_val_df = out.iloc[train_val_idx].copy()
    train_val_groups = groups.iloc[train_val_idx]
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_rel_idx, val_rel_idx = next(
        val_splitter.split(train_val_df, train_val_df["target"], groups=train_val_groups)
    )

    train_idx = train_val_df.index[train_rel_idx]
    val_idx = train_val_df.index[val_rel_idx]
    test_abs_idx = out.index[test_idx]

    out["split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_abs_idx, "split"] = "test"
    return out


def load_labeled_data(
    labels_csv_path: str = "data/labels.csv",
    force_regenerate_split: bool = True,
) -> pd.DataFrame:
    """Load labeled candidates and attach a validated duplicate-safe split."""
    out = pd.read_csv(labels_csv_path).copy()
    if "local_path" in out.columns:
        out["local_path"] = out["local_path"].apply(normalize_local_path)
    out["target"] = out["label"].map(LABEL_MAP)
    out = out[out["target"].isin([0, 1])].copy()

    has_saved_split = (
        "split" in out.columns
        and out["split"].fillna("").astype(str).str.strip().ne("").any()
    )
    if not force_regenerate_split and has_saved_split:
        out["split"] = out["split"].fillna("").astype(str).str.strip().str.lower()
    else:
        out = _assign_group_splits(out, random_state=42)

    validate_no_duplicate_leakage(out)
    return out


def validate_no_duplicate_leakage(df: pd.DataFrame) -> None:
    """Raise an error if any duplicate group appears in multiple splits."""
    split_groups = {
        split: set(build_group_id(df[df["split"] == split]).astype(str))
        for split in ["train", "val", "test"]
    }
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = (split_groups[left] & split_groups[right]) - {""}
        if overlap:
            raise ValueError(f"Duplicate leakage detected between {left} and {right}: {sorted(overlap)[:5]}")


def validate_no_page_leakage(df: pd.DataFrame) -> None:
    """Validate page holdout splits when that optional diagnostic is used."""
    split_groups = {
        split: set(build_page_group_id(df[df["split"] == split]).astype(str))
        for split in ["train", "val", "test"]
    }
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = (split_groups[left] & split_groups[right]) - {""}
        if overlap:
            raise ValueError(f"Page leakage detected between {left} and {right}: {sorted(overlap)[:5]}")


def build_split_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """Create the saved split manifest for the training notebook."""
    return pd.DataFrame(
        {
            "row_id": df.index.astype(str),
            "page_group": build_group_id(df).astype(str),
            "label": df.get("label", ""),
            "target": df.get("target", pd.Series(index=df.index, dtype=int)),
            "split": df.get("split", ""),
        }
    )


def build_model_pipeline(model_type: str = "logreg") -> Pipeline:
    """Build the metadata/URL baseline classification pipeline."""
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

    model = LogisticRegression(
        max_iter=1500,
        random_state=42,
        class_weight="balanced",
        C=0.7,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def predict_proba(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict probabilities of the `content` class."""
    feature_df = build_ml_feature_frame(df)
    return pd.Series(model.predict_proba(feature_df)[:, 1], index=df.index, name="ml_score")


def save_model_artifacts(artifacts: dict[str, Any], model_path: str = "models/best_model.pkl") -> None:
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, target)


def load_model_artifacts(model_path: str = "models/best_model.pkl") -> dict[str, Any]:
    return joblib.load(Path(model_path))


def train_and_save_model(
    labels_csv_path: str = "data/labels.csv",
    model_path: str = "models/best_model.pkl",
    model_type: str = "logreg",
) -> dict[str, Any]:
    """Train the model, select threshold on validation and save model artifacts."""
    df = load_labeled_data(labels_csv_path=labels_csv_path)
    features = build_ml_feature_frame(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    model = build_model_pipeline(model_type=model_type)
    model.fit(features.loc[train_df.index], train_df["target"])

    val_proba = pd.Series(model.predict_proba(features.loc[val_df.index])[:, 1], index=val_df.index)
    threshold, threshold_table = select_threshold_for_precision(
        val_df["target"],
        val_proba,
        min_positive_predictions=THRESHOLD_MIN_POSITIVE_PREDICTIONS,
        min_precision=THRESHOLD_MIN_PRECISION,
        tie_breaker=THRESHOLD_TIE_BREAKER,
    )

    train_proba = pd.Series(model.predict_proba(features.loc[train_df.index])[:, 1], index=train_df.index)
    train_pred = (train_proba >= threshold).astype(int)
    val_pred = (val_proba >= threshold).astype(int)
    test_proba = pd.Series(model.predict_proba(features.loc[test_df.index])[:, 1], index=test_df.index)
    test_pred = (test_proba >= threshold).astype(int)

    artifacts = {
        "model": model,
        "threshold": float(threshold),
        "model_type": model_type,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_strategy": "duplicate_safe_group_split(canonical_image_url->normalized_image_url->candidate_id)",
        "threshold_selection": {
            "min_precision": THRESHOLD_MIN_PRECISION,
            "min_positive_predictions": THRESHOLD_MIN_POSITIVE_PREDICTIONS,
            "tie_breaker": THRESHOLD_TIE_BREAKER,
        },
    }
    save_model_artifacts(artifacts=artifacts, model_path=model_path)

    return {
        "threshold": float(threshold),
        "train_metrics": evaluate_model_on_split(train_df["target"], train_pred, train_proba),
        "val_metrics": evaluate_model_on_split(val_df["target"], val_pred, val_proba),
        "test_metrics": evaluate_model_on_split(test_df["target"], test_pred, test_proba),
        "threshold_table": threshold_table,
        "split_assignment": build_split_assignment(df),
    }
