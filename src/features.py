from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import urlparse

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
