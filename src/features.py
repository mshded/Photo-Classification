from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import urlparse

import pandas as pd

# Conservative thresholds tuned for high precision
MIN_WIDTH = 120
MIN_HEIGHT = 120
MIN_AREA = 20_000
MAX_EXTREME_ASPECT_RATIO = 5.0
MIN_EXTREME_ASPECT_RATIO = 0.2
TRACKING_MAX_SIDE = 3
TRACKING_MAX_AREA = 16
TRACKING_MAX_FILE_SIZE_BYTES = 2_048
REPEATED_URL_THRESHOLD = 2

SUSPICIOUS_KEYWORDS = {
    "icon",
    "icons",
    "logo",
    "logos",
    "sprite",
    "sprites",
    "banner",
    "banners",
    "ads",
    "advert",
    "avatar",
    "placeholder",
    "pixel",
    "counter",
    "widget",
    "promo",
    "thumb",
}

TRACKING_PATTERNS = (
    "analytics",
    "counter",
    "track",
    "pixel",
    "metrics",
    "watch",
    "collect",
)


def normalize_text_for_match(text: str | None) -> str:
    if not text:
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def has_suspicious_keyword(*parts: str | None) -> bool:
    merged = normalize_text_for_match(" ".join([p or "" for p in parts]))
    if not merged:
        return False
    tokens = set(merged.split())
    return any(kw in tokens for kw in SUSPICIOUS_KEYWORDS)


def extract_url_flags(image_url: str, file_name: str = "", alt_text: str = "") -> Dict[str, Any]:
    normalized_url = normalize_text_for_match(image_url)
    normalized_file_name = normalize_text_for_match(file_name)
    normalized_alt = normalize_text_for_match(alt_text)
    parsed = urlparse(image_url) if image_url else urlparse("")

    has_tracking_hint = any(p in normalized_url for p in TRACKING_PATTERNS)
    has_suspicious = has_suspicious_keyword(image_url, file_name, alt_text)

    return {
        "has_suspicious_keyword": has_suspicious,
        "has_tracking_hint": has_tracking_hint,
        "url_path": parsed.path,
        "url_query": parsed.query,
        "normalized_file_name": normalized_file_name,
        "normalized_alt_text": normalized_alt,
    }


def is_probable_tracking_pixel(
    width: Any,
    height: Any,
    file_size_bytes: Any,
    image_url: str,
    domain: str = "",
) -> bool:
    try:
        w = float(width) if width is not None else None
        h = float(height) if height is not None else None
        fs = float(file_size_bytes) if file_size_bytes is not None else None
    except (TypeError, ValueError):
        w = h = fs = None

    tiny_geometry = bool(w and h and (w <= TRACKING_MAX_SIDE and h <= TRACKING_MAX_SIDE))
    tiny_area = bool(w and h and (w * h <= TRACKING_MAX_AREA))
    tiny_file = bool(fs is not None and fs <= TRACKING_MAX_FILE_SIZE_BYTES)
    tracking_url = any(p in normalize_text_for_match(f"{domain} {image_url}") for p in TRACKING_PATTERNS)

    return (tiny_geometry or tiny_area) and (tracking_url or tiny_file)


def is_too_small(width: Any, height: Any, area: Any) -> bool:
    try:
        w = float(width) if width is not None else None
        h = float(height) if height is not None else None
        a = float(area) if area is not None else None
    except (TypeError, ValueError):
        return False

    if w is not None and h is not None and (w < MIN_WIDTH or h < MIN_HEIGHT):
        return True
    if a is not None and a < MIN_AREA:
        return True
    return False


def has_extreme_aspect_ratio(aspect_ratio: Any) -> bool:
    try:
        ar = float(aspect_ratio)
    except (TypeError, ValueError):
        return False

    return ar > MAX_EXTREME_ASPECT_RATIO or ar < MIN_EXTREME_ASPECT_RATIO


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "t"})


def build_ml_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare tabular ML features using shared logic for train and inference."""
    work_df = df.copy()

    numeric_cols = ["width", "height", "area", "aspect_ratio", "file_size_bytes"]
    for col in numeric_cols:
        if col not in work_df.columns:
            work_df[col] = pd.NA
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    for col in ["format", "image_url", "file_name", "alt_text", "domain"]:
        if col not in work_df.columns:
            work_df[col] = ""
        work_df[col] = work_df[col].fillna("").astype(str)

    if "is_tiny" in work_df.columns:
        is_tiny = _to_bool_series(work_df["is_tiny"])
    else:
        is_tiny = work_df.apply(
            lambda r: is_too_small(r.get("width"), r.get("height"), r.get("area")), axis=1
        )

    if "has_ui_keyword" in work_df.columns:
        has_ui_keyword = _to_bool_series(work_df["has_ui_keyword"])
    else:
        has_ui_keyword = work_df.apply(
            lambda r: has_suspicious_keyword(r.get("image_url", ""), r.get("file_name", ""), r.get("alt_text", "")),
            axis=1,
        )

    if "is_suspicious_domain" in work_df.columns:
        is_suspicious_domain = _to_bool_series(work_df["is_suspicious_domain"])
    else:
        is_suspicious_domain = work_df["domain"].str.contains("analytics|ad|tracker|pixel", case=False, regex=True)

    base = pd.DataFrame(
        {
            "width": work_df["width"],
            "height": work_df["height"],
            "area": work_df["area"],
            "aspect_ratio": work_df["aspect_ratio"],
            "file_size_bytes": work_df["file_size_bytes"],
            "format": work_df["format"].str.lower().replace("", "unknown"),
            "is_tiny": is_tiny.astype(int),
            "is_suspicious_domain": is_suspicious_domain.astype(int),
            "has_ui_keyword": has_ui_keyword.astype(int),
        }
    )

    base["is_too_small"] = base.apply(
        lambda r: int(is_too_small(r["width"], r["height"], r["area"])), axis=1
    )
    base["has_extreme_aspect_ratio"] = base["aspect_ratio"].apply(
        lambda x: int(has_extreme_aspect_ratio(x))
    )

    tracking_flags = work_df.apply(
        lambda r: extract_url_flags(r.get("image_url", ""), r.get("file_name", ""), r.get("alt_text", "")),
        axis=1,
    )
    base["has_tracking_hint"] = tracking_flags.apply(lambda x: int(bool(x["has_tracking_hint"])))
    base["has_suspicious_keyword"] = tracking_flags.apply(
        lambda x: int(bool(x["has_suspicious_keyword"]))
    )

    return base