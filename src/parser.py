from __future__ import annotations

from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

IMAGE_ATTRS = [
    "src",
    "data-src",
    "data-lazy-src",
    "data-original",
    "data-image",
    "data-fallback-src",
    "data-src-retina",
    "data-lazy",
]

SRCSET_ATTRS = [
    "srcset",
    "data-srcset",
]

def is_empty_or_bad_url(url: Optional[str]) -> bool:
    if url is None:
        return True

    url = str(url).strip()
    if not url:
        return True

    bad_prefixes = ("data:", "blob:", "javascript:", "mailto:", "tel:")
    if url.startswith(bad_prefixes):
        return True

    return False

def fetch_page_html(url: str, timeout: tuple[int, int] = (10, 30)) -> str:
    response = requests.get(
        url,
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response.text

def parse_srcset(srcset_string: Optional[str]) -> List[str]:
    if srcset_string is None:
        return []

    srcset_string = srcset_string.strip()
    if not srcset_string:
        return []

    result = []
    parts = srcset_string.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        url_part = part.split()[0].strip()
        if url_part and not is_empty_or_bad_url(url_part):
            result.append(url_part)

    return result

def normalize_image_url(image_url: Optional[str], page_url: str) -> Optional[str]:
    if is_empty_or_bad_url(image_url):
        return None

    image_url = image_url.strip()

    # //cdn.site.com/img.jpg
    if image_url.startswith("//"):
        parsed_page = urlparse(page_url)
        scheme = parsed_page.scheme if parsed_page.scheme else "https"
        return f"{scheme}:{image_url}"

    return urljoin(page_url, image_url)

def extract_img_candidates_from_html(html: str, page_url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[Dict] = []

    # 1. Обычные img
    img_tags = soup.find_all("img")

    for tag_index, tag in enumerate(img_tags):
        alt_text = (tag.get("alt") or "").strip()
        width_attr = (tag.get("width") or "").strip()
        height_attr = (tag.get("height") or "").strip()

        # Извлекаем одиночные URL-атрибуты
        for attr_name in IMAGE_ATTRS:
            raw_url = tag.get(attr_name)
            normalized_url = normalize_image_url(raw_url, page_url)

            if normalized_url is None:
                continue

            candidates.append(
                {
                    "page_url": page_url,
                    "image_url": normalized_url,
                    "source_attr": attr_name,
                    "alt_text": alt_text,
                    "tag_index": tag_index,
                    "width_attr": width_attr,
                    "height_attr": height_attr,
                }
            )

        # Извлекаем srcset-подобные атрибуты
        for attr_name in SRCSET_ATTRS:
            srcset_value = tag.get(attr_name)
            for raw_url in parse_srcset(srcset_value):
                normalized_url = normalize_image_url(raw_url, page_url)
                if normalized_url is None:
                    continue

                candidates.append(
                    {
                        "page_url": page_url,
                        "image_url": normalized_url,
                        "source_attr": attr_name,
                        "alt_text": alt_text,
                        "tag_index": tag_index,
                        "width_attr": width_attr,
                        "height_attr": height_attr,
                    }
                )

    # 2. picture > source
    source_tags = soup.find_all("source")

    for tag_index, tag in enumerate(source_tags):
        for attr_name in SRCSET_ATTRS:
            srcset_value = tag.get(attr_name)
            for raw_url in parse_srcset(srcset_value):
                normalized_url = normalize_image_url(raw_url, page_url)
                if normalized_url is None:
                    continue

                candidates.append(
                    {
                        "page_url": page_url,
                        "image_url": normalized_url,
                        "source_attr": f"source:{attr_name}",
                        "alt_text": "",
                        "tag_index": tag_index,
                        "width_attr": "",
                        "height_attr": "",
                    }
                )

    return candidates

def deduplicate_candidates(candidates: List[Dict]) -> List[Dict]:
    unique_candidates = []
    seen = set()

    for item in candidates:
        key = (
            item.get("page_url", ""),
            item.get("image_url", ""),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(item)

    return unique_candidates

def collect_image_candidates(
    url: str,
    save_csv_path: Optional[str] = None,
    save_json_path: Optional[str] = None,
) -> pd.DataFrame:
    html = fetch_page_html(url)

    candidates = extract_img_candidates_from_html(html, url)
    candidates = deduplicate_candidates(candidates)

    df = pd.DataFrame(candidates)

    if not df.empty:
        df["domain"] = df["image_url"].apply(lambda x: urlparse(x).netloc)
        df["file_name"] = df["image_url"].apply(
            lambda x: urlparse(x).path.split("/")[-1] if urlparse(x).path else ""
        )
        df["candidate_id"] = [
            f"cand_{i:06d}" for i in range(len(df))
        ]
    else:
        df = pd.DataFrame(
            columns=[
                "page_url",
                "image_url",
                "source_attr",
                "alt_text",
                "tag_index",
                "width_attr",
                "height_attr",
                "domain",
                "file_name",
                "candidate_id",
            ]
        )

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False, encoding="utf-8")

    if save_json_path is not None:
        df.to_json(save_json_path, orient="records", force_ascii=False, indent=2)

    return df

