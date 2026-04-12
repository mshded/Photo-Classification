from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import urlparse

import pandas as pd

MIN_WIDTH = 120
MIN_HEIGHT = 120
MIN_AREA = 20_000
TINY_MAX_SIDE = 5
TINY_MAX_AREA = 25
MAX_EXTREME_ASPECT_RATIO = 5.0
MIN_EXTREME_ASPECT_RATIO = 0.2
TRACKING_MAX_SIDE = 3
TRACKING_MAX_AREA = 16
TRACKING_MAX_FILE_SIZE_BYTES = 2_048
REPEATED_URL_THRESHOLD = 2

# Эти keywords используются как мягкие сигналы для модели.
# Они не должны автоматически убивать candidate на этапе hard prefilter.
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

# Эти keywords уже достаточно надёжны для hard prefilter.
# Набор сделан уже, чем SUSPICIOUS_KEYWORDS, чтобы не убивать recall.
HARD_BLOCK_KEYWORDS = {
    "icon",
    "icons",
    "logo",
    "logos",
    "sprite",
    "sprites",
    "pixel",
    "counter",
    "analytics",
    "tracking",
    "tracker",
    "doubleclick",
    "googletagmanager",
    "gtm",
}

TRACKING_PATTERNS = (
    "analytics",
    "counter",
    "track",
    "tracking",
    "pixel",
    "metrics",
    "watch",
    "collect",
    "gtm",
    "googletagmanager",
    "doubleclick",
    "mc.yandex",
    "tns-counter",
)


def normalize_text_for_match(text: str | None) -> str:
    if not text:
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _has_any_keyword(keywords: set[str], *parts: str | None) -> bool:
    merged = normalize_text_for_match(" ".join([p or "" for p in parts]))
    if not merged:
        return False
    tokens = set(merged.split())
    return any(kw in tokens for kw in keywords)


def has_suspicious_keyword(*parts: str | None) -> bool:
    return _has_any_keyword(SUSPICIOUS_KEYWORDS, *parts)


def has_hard_block_keyword(*parts: str | None) -> bool:
    return _has_any_keyword(HARD_BLOCK_KEYWORDS, *parts)


def extract_url_flags(image_url: str, file_name: str = "", alt_text: str = "") -> Dict[str, Any]:
    normalized_url = normalize_text_for_match(image_url)
    normalized_file_name = normalize_text_for_match(file_name)
    normalized_alt = normalize_text_for_match(alt_text)
    parsed = urlparse(image_url) if image_url else urlparse("")

    has_tracking_hint = any(p in normalized_url for p in TRACKING_PATTERNS)
    has_suspicious = has_suspicious_keyword(image_url, file_name, alt_text)
    has_hard_block = has_hard_block_keyword(image_url, file_name)

    return {
        "has_suspicious_keyword": has_suspicious,
        "has_tracking_hint": has_tracking_hint,
        "has_hard_block_keyword": has_hard_block,
        "url_path": parsed.path,
        "url_query": parsed.query,
        "normalized_file_name": normalized_file_name,
        "normalized_alt_text": normalized_alt,
    }


def has_analytics_url_hint(image_url: str, domain: str = "") -> bool:
    haystack = normalize_text_for_match(f"{domain} {image_url}")
    if not haystack:
        return False
    return any(token in haystack for token in TRACKING_PATTERNS)


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


def is_tiny_image(width: Any, height: Any, area: Any) -> bool:
    try:
        w = float(width) if width is not None else None
        h = float(height) if height is not None else None
        a = float(area) if area is not None else None
    except (TypeError, ValueError):
        return False

    if w is not None and h is not None and w <= TINY_MAX_SIDE and h <= TINY_MAX_SIDE:
        return True
    if a is not None and a <= TINY_MAX_AREA:
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
    work_df = df.copy()

    numeric_cols = ["width", "height", "area", "aspect_ratio", "file_size_bytes"]
    for col in numeric_cols:
        if col not in work_df.columns:
            work_df[col] = pd.NA
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    for col in ["format", "image_url", "file_name", "alt_text", "domain", "source_attr"]:
        if col not in work_df.columns:
            work_df[col] = ""
        work_df[col] = work_df[col].fillna("").astype(str)

    is_tiny = work_df.apply(
        lambda r: is_tiny_image(r.get("width"), r.get("height"), r.get("area")), axis=1
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

    tracking_flags = work_df.apply(
        lambda r: extract_url_flags(r.get("image_url", ""), r.get("file_name", ""), r.get("alt_text", "")),
        axis=1,
    )

    repeated_url_counts = work_df["image_url"].fillna("").map(work_df["image_url"].fillna("").value_counts())
    alt_text_len = work_df["alt_text"].str.len().fillna(0)
    file_name_len = work_df["file_name"].str.len().fillna(0)
    url_depth = work_df["image_url"].apply(lambda x: len([p for p in urlparse(x).path.split("/") if p]))

    base = pd.DataFrame(
        {
            "width": work_df["width"],
            "height": work_df["height"],
            "area": work_df["area"],
            "aspect_ratio": work_df["aspect_ratio"],
            "file_size_bytes": work_df["file_size_bytes"],
            "format": work_df["format"].str.lower().replace("", "unknown"),
            "source_attr": work_df["source_attr"].replace("", "unknown"),
            "is_tiny": is_tiny.astype(int),
            "is_suspicious_domain": is_suspicious_domain.astype(int),
            "has_ui_keyword": has_ui_keyword.astype(int),
            "has_tracking_hint": tracking_flags.apply(lambda x: int(bool(x["has_tracking_hint"]))),
            "has_suspicious_keyword": tracking_flags.apply(lambda x: int(bool(x["has_suspicious_keyword"]))),
            "has_hard_block_keyword": tracking_flags.apply(lambda x: int(bool(x["has_hard_block_keyword"]))),
            "repeated_url_count": repeated_url_counts.fillna(0).astype(float),
            "alt_text_length": alt_text_len.astype(float),
            "file_name_length": file_name_len.astype(float),
            "url_depth": url_depth.astype(float),
        }
    )

    base["is_too_small"] = base.apply(
        lambda r: int(is_too_small(r["width"], r["height"], r["area"])), axis=1
    )
    base["has_extreme_aspect_ratio"] = base["aspect_ratio"].apply(
        lambda x: int(has_extreme_aspect_ratio(x))
    )
    base["is_large_image"] = ((base["width"] >= 240) & (base["height"] >= 240) & (base["area"] >= 60_000)).astype(int)
    base["has_descriptive_alt"] = (base["alt_text_length"] >= 10).astype(int)

    return base
