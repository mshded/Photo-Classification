from __future__ import annotations

import hashlib
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
]
CATEGORICAL_FEATURES = ["format"]
SIZE_QUERY_KEYS = {
    "w",
    "h",
    "width",
    "height",
    "size",
    "sz",
    "dpr",
    "quality",
    "q",
    "crop",
    "fit",
    "resize",
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
    if not local_path or pd.isna(local_path):
        return ""

    text = str(local_path).strip().replace("\\", "/")
    lowered = text.lower()
    marker = "/data/"
    marker_idx = lowered.find(marker)
    if marker_idx >= 0:
        text = text[marker_idx + 1 :]

    cleaned = Path(text)
    try:
        if cleaned.is_absolute():
            return cleaned.resolve().relative_to(_project_root().resolve()).as_posix()
    except Exception:
        return cleaned.as_posix()
    return cleaned.as_posix()


def _to_existing_path(local_path: str | Path | None) -> Path | None:
    normalized = normalize_local_path(local_path)
    if not normalized:
        return None
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = _project_root() / candidate
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _content_hash_from_local_path(local_path: str) -> str | None:
    path = _to_existing_path(local_path)
    if path is None:
        return None

    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_for_grouping(image_url: str) -> str:
    if not image_url:
        return ""
    parsed = urlsplit(str(image_url).strip())
    path = parsed.path or ""
    path = RESIZE_TOKEN_RE.sub("", path)
    path = re.sub(r"/{2,}", "/", path)
    path = EXT_RE.sub("", path)
    path = path.rstrip("/")

    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in SIZE_QUERY_KEYS:
            continue
        query_pairs.append((key.lower(), value))
    query = urlencode(sorted(query_pairs))
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path.lower(), query, ""))


def build_group_id(df: pd.DataFrame) -> pd.Series:
    content_hash = df.get("local_path", pd.Series(index=df.index, dtype="object")).apply(_content_hash_from_local_path)

    canonical_url = df.get("image_url", pd.Series(index=df.index, dtype="object")).fillna("").astype(str).apply(
        _canonicalize_for_grouping
    )
    normalized_url = df.get("image_url", pd.Series(index=df.index, dtype="object")).fillna("").astype(str).apply(
        _normalize_image_url
    )
    page_url = df.get("page_url", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)

    group_id = content_hash.copy()
    group_id = group_id.fillna("")
    group_id = group_id.where(group_id.str.len() > 0, canonical_url)
    group_id = group_id.where(group_id.str.len() > 0, normalized_url)
    group_id = group_id.where(group_id.str.len() > 0, page_url)
    group_id = group_id.where(group_id.str.len() > 0, df.index.astype(str))
    return group_id.astype(str)


def _assign_group_splits(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    out = df.copy()
    groups = build_group_id(out)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(out, out["target"], groups=groups))

    train_val_df = out.iloc[train_val_idx].copy()
    train_val_groups = groups.iloc[train_val_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_rel_idx, val_rel_idx = next(
        gss_val.split(train_val_df, train_val_df["target"], groups=train_val_groups)
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
    df = pd.read_csv(labels_csv_path)
    out = df.copy()
    if "local_path" in out.columns:
        out["local_path"] = out["local_path"].apply(normalize_local_path)
    out["target"] = out["label"].map(LABEL_MAP)
    out = out[out["target"].isin([0, 1])].copy()

    if (
        not force_regenerate_split
        and "split" in out.columns
        and out["split"].fillna("").str.strip().ne("").any()
    ):
        out["split"] = out["split"].fillna("").astype(str).str.strip().str.lower()
        return out

    return _assign_group_splits(out, random_state=42)


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

    train_pred = (
        pd.Series(model.predict_proba(features.loc[train_df.index])[:, 1], index=train_df.index) >= threshold
    ).astype(int)
    val_pred = (val_proba >= threshold).astype(int)
    test_proba = pd.Series(model.predict_proba(features.loc[test_df.index])[:, 1], index=test_df.index)
    test_pred = (test_proba >= threshold).astype(int)

    artifacts = {
        "model": model,
        "threshold": float(threshold),
        "model_type": model_type,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_strategy": "group_split(content_hash->canonical_image_id->normalized_image_url->page_url)",
    }
    save_model_artifacts(artifacts=artifacts, model_path=model_path)

    return {
        "threshold": float(threshold),
        "train_metrics": evaluate_model_on_split(train_df["target"], train_pred),
        "val_metrics": evaluate_model_on_split(val_df["target"], val_pred, val_proba),
        "test_metrics": evaluate_model_on_split(test_df["target"], test_pred, test_proba),
        "threshold_table": threshold_table,
    }
