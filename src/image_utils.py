from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List
from urllib.parse import urlparse
import hashlib

import requests
from PIL import Image
import matplotlib.pyplot as plt

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

VALID_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"
}


def make_unique_filename(image_url: str, candidate_id: str) -> str:
    parsed = urlparse(image_url)
    ext = Path(parsed.path).suffix.lower()

    if ext not in VALID_IMAGE_EXTENSIONS:
        ext = ".img"

    url_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:10]
    return f"{candidate_id}_{url_hash}{ext}"


def download_image(
    image_url: str,
    save_path: Path,
    timeout: int = 25,
) -> tuple[bool, str]:
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(
            image_url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True, ""

    except Exception as e:
        return False, str(e)


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


def is_valid_image(path: Path) -> bool:
    img = safe_open_image(path)
    return img is not None


def compute_basic_geometry(width: Optional[int], height: Optional[int]) -> Dict[str, Optional[float]]:
    if width is None or height is None or height == 0:
        return {
            "area": None,
            "aspect_ratio": None,
        }

    return {
        "area": int(width * height),
        "aspect_ratio": float(width / height),
    }


def get_image_metadata(path: Path) -> Dict:
    img = safe_open_image(path)

    if img is None:
        return {
            "is_valid_image": False,
            "width": None,
            "height": None,
            "format": None,
            "mode": None,
            "file_size_bytes": path.stat().st_size if path.exists() else None,
            "area": None,
            "aspect_ratio": None,
            "image_error": "Cannot open image",
        }

    width, height = img.size
    basic_geometry = compute_basic_geometry(width, height)

    return {
        "is_valid_image": True,
        "width": width,
        "height": height,
        "format": img.format,
        "mode": img.mode,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "area": basic_geometry["area"],
        "aspect_ratio": basic_geometry["aspect_ratio"],
        "image_error": "",
    }


def show_images_grid(paths: List[str], n: int = 9, figsize: tuple = (12, 12)) -> None:
    paths = [p for p in paths if p][:n]
    if not paths:
        print("No images to display.")
        return

    cols = 3
    rows = (len(paths) + cols - 1) // cols

    plt.figure(figsize=figsize)

    for i, path in enumerate(paths, start=1):
        plt.subplot(rows, cols, i)
        img = safe_open_image(Path(path))
        if img is not None:
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "Invalid image", ha="center", va="center")
        plt.axis("off")
        plt.title(Path(path).name[:40])

    plt.tight_layout()
    plt.show()