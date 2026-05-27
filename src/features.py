from __future__ import annotations

import re
from typing import Any
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

# Мягкие признаки: используются моделью, но сами по себе не удаляют candidate.
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

# Надёжные technical/UI-сигналы для hard prefilter.
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


def safe_str(value: Any) -> str:
    """Convert None/NaN to an empty string and other values to text."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def normalize_text_for_match(text: Any) -> str:
    text = safe_str(text)
    if not text:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(normalized.split())


def _has_any_keyword(keywords: set[str], *parts: Any) -> bool:
    merged = normalize_text_for_match(" ".join(safe_str(part) for part in parts))
    if not merged:
        return False
    tokens = set(merged.split())
    return any(keyword in tokens for keyword in keywords)


def has_suspicious_keyword(*parts: Any) -> bool:
    return _has_any_keyword(SUSPICIOUS_KEYWORDS, *parts)


def has_hard_block_keyword(*parts: Any) -> bool:
    return _has_any_keyword(HARD_BLOCK_KEYWORDS, *parts)


def extract_url_flags(image_url: Any, file_name: Any = "", alt_text: Any = "") -> dict[str, Any]:
    """Extract keyword and tracking-related flags from candidate metadata."""
    url_text = safe_str(image_url)
    normalized_url = normalize_text_for_match(url_text)
    normalized_file_name = normalize_text_for_match(file_name)
    normalized_alt = normalize_text_for_match(alt_text)
    parsed = urlparse(url_text)

    return {
        "has_suspicious_keyword": has_suspicious_keyword(url_text, file_name, alt_text),
        "has_tracking_hint": any(pattern in normalized_url for pattern in TRACKING_PATTERNS),
        "has_hard_block_keyword": has_hard_block_keyword(url_text, file_name),
        "url_path": parsed.path,
        "url_query": parsed.query,
        "normalized_file_name": normalized_file_name,
        "normalized_alt_text": normalized_alt,
    }


def has_analytics_url_hint(image_url: Any, domain: Any = "") -> bool:
    haystack = normalize_text_for_match(f"{safe_str(domain)} {safe_str(image_url)}")
    return bool(haystack) and any(token in haystack for token in TRACKING_PATTERNS)


def is_probable_tracking_pixel(
    width: Any,
    height: Any,
    file_size_bytes: Any,
    image_url: Any,
    domain: Any = "",
) -> bool:
    try:
        w = float(width) if width is not None and pd.notna(width) else None
        h = float(height) if height is not None and pd.notna(height) else None
        fs = float(file_size_bytes) if file_size_bytes is not None and pd.notna(file_size_bytes) else None
    except (TypeError, ValueError):
        w = h = fs = None

    tiny_geometry = bool(w is not None and h is not None and w <= TRACKING_MAX_SIDE and h <= TRACKING_MAX_SIDE)
    tiny_area = bool(w is not None and h is not None and w * h <= TRACKING_MAX_AREA)
    tiny_file = bool(fs is not None and fs <= TRACKING_MAX_FILE_SIZE_BYTES)
    tracking_url = has_analytics_url_hint(image_url, domain)
    return (tiny_geometry or tiny_area) and (tracking_url or tiny_file)


def is_too_small(width: Any, height: Any, area: Any) -> bool:
    try:
        w = float(width) if width is not None and pd.notna(width) else None
        h = float(height) if height is not None and pd.notna(height) else None
        a = float(area) if area is not None and pd.notna(area) else None
    except (TypeError, ValueError):
        return False

    return bool(
        (w is not None and h is not None and (w < MIN_WIDTH or h < MIN_HEIGHT))
        or (a is not None and a < MIN_AREA)
    )


def is_tiny_image(width: Any, height: Any, area: Any) -> bool:
    try:
        w = float(width) if width is not None and pd.notna(width) else None
        h = float(height) if height is not None and pd.notna(height) else None
        a = float(area) if area is not None and pd.notna(area) else None
    except (TypeError, ValueError):
        return False

    return bool(
        (w is not None and h is not None and w <= TINY_MAX_SIDE and h <= TINY_MAX_SIDE)
        or (a is not None and a <= TINY_MAX_AREA)
    )


def has_extreme_aspect_ratio(aspect_ratio: Any) -> bool:
    try:
        ratio = float(aspect_ratio)
    except (TypeError, ValueError):
        return False
    return ratio > MAX_EXTREME_ASPECT_RATIO or ratio < MIN_EXTREME_ASPECT_RATIO


def build_ml_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the stable metadata/URL feature table used by the classifier."""
    work_df = df.copy()

    for column in ["width", "height", "area", "aspect_ratio", "file_size_bytes"]:
        if column not in work_df.columns:
            work_df[column] = pd.NA
        work_df[column] = pd.to_numeric(work_df[column], errors="coerce")

    for column in ["format", "image_url", "file_name", "alt_text", "domain", "source_attr"]:
        if column not in work_df.columns:
            work_df[column] = ""
        work_df[column] = work_df[column].apply(safe_str)

    tracking_flags = work_df.apply(
        lambda row: extract_url_flags(
            row.get("image_url", ""), row.get("file_name", ""), row.get("alt_text", "")
        ),
        axis=1,
    )
    repeated_url_counts = work_df["image_url"].map(work_df["image_url"].value_counts())

    base = pd.DataFrame(
        {
            "width": work_df["width"],
            "height": work_df["height"],
            "area": work_df["area"],
            "aspect_ratio": work_df["aspect_ratio"],
            "file_size_bytes": work_df["file_size_bytes"],
            "format": work_df["format"].str.lower().replace("", "unknown"),
            "source_attr": work_df["source_attr"].replace("", "unknown"),
            "is_tiny": work_df.apply(
                lambda row: int(is_tiny_image(row.get("width"), row.get("height"), row.get("area"))), axis=1
            ),
            "is_suspicious_domain": work_df.apply(
                lambda row: int(has_analytics_url_hint(row.get("image_url", ""), row.get("domain", ""))), axis=1
            ),
            "has_ui_keyword": tracking_flags.apply(lambda flags: int(bool(flags["has_suspicious_keyword"]))),
            "has_tracking_hint": tracking_flags.apply(lambda flags: int(bool(flags["has_tracking_hint"]))),
            "has_suspicious_keyword": tracking_flags.apply(lambda flags: int(bool(flags["has_suspicious_keyword"]))),
            "has_hard_block_keyword": tracking_flags.apply(lambda flags: int(bool(flags["has_hard_block_keyword"]))),
            "repeated_url_count": repeated_url_counts.fillna(0).astype(float),
            "alt_text_length": work_df["alt_text"].str.len().fillna(0).astype(float),
            "file_name_length": work_df["file_name"].str.len().fillna(0).astype(float),
            "url_depth": work_df["image_url"].apply(
                lambda value: len([part for part in urlparse(value).path.split("/") if part])
            ).astype(float),
        }
    )

    base["is_too_small"] = base.apply(
        lambda row: int(is_too_small(row["width"], row["height"], row["area"])), axis=1
    )
    base["has_extreme_aspect_ratio"] = base["aspect_ratio"].apply(
        lambda value: int(has_extreme_aspect_ratio(value))
    )
    base["is_large_image"] = (
        (base["width"] >= 240) & (base["height"] >= 240) & (base["area"] >= 60_000)
    ).astype(int)
    base["has_descriptive_alt"] = (base["alt_text_length"] >= 10).astype(int)
    return base
