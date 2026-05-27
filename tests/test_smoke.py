from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import _assign_group_splits, build_group_id
from src.features import build_ml_feature_frame, is_probable_tracking_pixel
from src.metrics import compute_classification_metrics
from src.parser import deduplicate_candidates, extract_img_candidates_from_html


def test_metrics_smoke() -> None:
    metrics = compute_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert abs(metrics["precision"] - 0.5) < 1e-9


def test_parser_extracts_common_image_sources() -> None:
    html = """
    <html><body>
        <img src="/a.jpg" alt="a">
        <img data-src="/b.jpg" srcset="/c.jpg 1x, /d.jpg 2x">
        <picture><source srcset="/e.jpg 1x"></picture>
    </body></html>
    """
    candidates = extract_img_candidates_from_html(html, "https://example.com/p")
    urls = {item["image_url"] for item in deduplicate_candidates(candidates)}
    assert "https://example.com/a.jpg" in urls
    assert "https://example.com/b.jpg" in urls
    assert "https://example.com/c.jpg" in urls
    assert "https://example.com/e.jpg" in urls


def test_features_handle_nan_and_detect_tracking_pixel() -> None:
    df = pd.DataFrame(
        [
            {
                "image_url": None,
                "file_name": float("nan"),
                "alt_text": None,
                "domain": None,
                "source_attr": None,
                "width": 1,
                "height": 1,
                "area": 1,
                "aspect_ratio": 1.0,
                "file_size_bytes": 100,
                "format": "png",
            }
        ]
    )
    assert len(build_ml_feature_frame(df)) == 1
    assert is_probable_tracking_pixel(1, 1, 100, "https://x/track/pixel", "x")


def test_duplicate_url_variants_stay_in_one_split() -> None:
    df = pd.DataFrame(
        {
            "target": [1, 1, 0, 0, 1, 0],
            "image_url": [
                "https://site.org/image.jpg?w=100",
                "https://site.org/image.jpg?w=500",
                "https://site.org/logo.png",
                "https://site.org/banner.png",
                "https://site.org/photo2.jpg",
                "https://site.org/icon.png",
            ],
        }
    )
    split_df = _assign_group_splits(df, random_state=42)
    assert split_df.iloc[[0, 1]]["split"].nunique() == 1


def test_group_id_does_not_depend_on_local_raw_files(tmp_path: Path) -> None:
    existing_file = tmp_path / "downloaded_image.jpg"
    existing_file.write_bytes(b"some binary image bytes")
    df = pd.DataFrame(
        {
            "candidate_id": ["img_small", "img_large"],
            "image_url": [
                "https://cdn.example.org/products/shoes/main.jpg?w=200&q=60",
                "https://cdn.example.org/products/shoes/main.webp?w=800&q=90",
            ],
            "local_path": [str(existing_file), "data/raw/missing_image.webp"],
        }
    )
    group_ids = build_group_id(df)
    assert group_ids.iloc[0] == group_ids.iloc[1]
