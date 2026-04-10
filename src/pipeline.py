from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd

from src.features import (
    REPEATED_URL_THRESHOLD,
    extract_url_flags,
    has_extreme_aspect_ratio,
    is_probable_tracking_pixel,
)
from src.image_utils import download_image, get_image_metadata, make_unique_filename
from src.parser import collect_image_candidates

FINAL_KEEP_COLUMNS = [
    "candidate_id",
    "image_url",
    "local_path",
    "hard_prefilter_keep",
    "hard_reject_reason",
    "ml_score",
    "ml_pred",
    "final_keep",
]


def make_page_id(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.strip("/").replace("/", "_") or "home"
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)[:50]
    host = parsed.netloc.replace(":", "_")
    return f"{host}_{slug}".strip("_")


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def write_page_info(
    page_url: str,
    page_id: str,
    model_path: str,
    raw_dir: Path,
    output_dir: Path,
) -> Dict[str, str]:
    payload = {
        "page_url": page_url,
        "page_id": page_id,
        "pipeline": "ml_with_hard_prefilter",
        "model_path": model_path,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "page_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def prepare_candidates_dataframe(url: str, raw_dir: Path) -> pd.DataFrame:
    df = collect_image_candidates(url)
    if df.empty:
        return df

    page_id = make_page_id(url)
    raw_page_dir = raw_dir / page_id

    df = df.copy()
    df["page_id"] = page_id
    df["raw_page_dir"] = str(raw_page_dir)
    df["candidate_id"] = [f"{page_id}_cand_{i:06d}" for i in range(len(df))]
    df["local_file_name"] = df.apply(lambda row: make_unique_filename(row["image_url"], row["candidate_id"]), axis=1)
    df["local_path"] = df["local_file_name"].apply(lambda f: str(raw_page_dir / f))
    return df


def download_candidates(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    statuses: List[Dict[str, str]] = []

    for _, row in out.iterrows():
        local_path = Path(row["local_path"])
        ok, err = download_image(row["image_url"], local_path)
        statuses.append({"download_ok": ok, "download_error": err})

    status_df = pd.DataFrame(statuses)
    return pd.concat([out.reset_index(drop=True), status_df], axis=1)


def enrich_with_image_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    records: List[Dict] = []

    for _, row in out.iterrows():
        if not row.get("download_ok", False):
            records.append(
                {
                    "is_valid_image": False,
                    "width": None,
                    "height": None,
                    "format": None,
                    "mode": None,
                    "file_size_bytes": None,
                    "area": None,
                    "aspect_ratio": None,
                    "image_error": row.get("download_error", "download_failed"),
                }
            )
            continue

        meta = get_image_metadata(Path(row["local_path"]))
        records.append(meta)

    return pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def apply_hard_prefilter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    repeat_counts = Counter(out["image_url"].fillna(""))

    keeps = []
    reasons = []
    flags_col = []

    for _, row in out.iterrows():
        flags = []

        if not row.get("download_ok", False):
            flags.append("download_failed")
        if not row.get("is_valid_image", False):
            flags.append("invalid_image")

        width = row.get("width")
        height = row.get("height")
        area = row.get("area")
        file_size_bytes = row.get("file_size_bytes")

        try:
            w = float(width) if width is not None else None
            h = float(height) if height is not None else None
        except (TypeError, ValueError):
            w = h = None

        if w is not None and h is not None and w <= 5 and h <= 5:
            flags.append("tiny_dimensions_le_5")

        if file_size_bytes is not None and pd.notna(file_size_bytes):
            try:
                if float(file_size_bytes) <= 512:
                    flags.append("tiny_file_size")
            except (TypeError, ValueError):
                pass

        url_flags = extract_url_flags(
            row.get("image_url", ""),
            row.get("file_name", ""),
            row.get("alt_text", ""),
        )
        if url_flags["has_tracking_hint"]:
            flags.append("tracking_url_hint")
        if url_flags["has_suspicious_keyword"]:
            flags.append("ui_or_ads_keyword")

        if is_probable_tracking_pixel(
            width,
            height,
            file_size_bytes,
            row.get("image_url", ""),
            row.get("domain", ""),
        ):
            flags.append("probable_tracking_pixel")

        soft_flags = []
        if has_extreme_aspect_ratio(row.get("aspect_ratio")):
            soft_flags.append("extreme_aspect_ratio")

        image_url = row.get("image_url", "")
        if repeat_counts.get(image_url, 0) >= REPEATED_URL_THRESHOLD:
            soft_flags.append("repeated_url")
        if url_flags["has_suspicious_keyword"]:
            soft_flags.append("ui_or_ads_keyword")

        hard_reject = {
            "download_failed",
            "invalid_image",
            "tiny_dimensions_le_5",
            "tiny_file_size",
            "probable_tracking_pixel",
            "tracking_url_hint",
        }

        keep = not any(f in hard_reject for f in flags)
        keeps.append(keep)
        reasons.append("" if keep else ";".join([f for f in flags if f in hard_reject]))
        flags_col.append(json.dumps(flags + soft_flags, ensure_ascii=False))

    out["hard_prefilter_keep"] = keeps
    out["hard_reject_reason"] = reasons
    out["hard_rule_flags"] = flags_col
    return out


def apply_ml_filter(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    from src.classifier import load_model_artifacts, predict_proba

    out = df.copy()
    out["ml_score"] = pd.NA
    out["ml_pred"] = 0

    ml_candidates_mask = out["hard_prefilter_keep"].fillna(False)
    if not ml_candidates_mask.any():
        out["final_keep"] = False
        return out

    artifacts = load_model_artifacts(model_path)
    model = artifacts["model"]
    threshold = float(artifacts.get("threshold", 0.5))

    ml_df = out.loc[ml_candidates_mask].copy()
    scores = predict_proba(model, ml_df)
    preds = (scores >= threshold).astype(int)

    out.loc[ml_candidates_mask, "ml_score"] = scores.values
    out.loc[ml_candidates_mask, "ml_pred"] = preds.values
    out["final_keep"] = out["hard_prefilter_keep"].fillna(False) & (out["ml_pred"] == 1)
    return out


def summarize_pipeline_results(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "total_candidates": 0,
            "downloaded_ok": 0,
            "hard_prefilter_rejected": 0,
            "ml_candidates": 0,
            "final_kept": 0,
            "top_reject_reasons": {},
        }

    rejected = df.loc[~df["hard_prefilter_keep"], "hard_reject_reason"].fillna("")
    exploded = (
        rejected.str.split(";").explode().str.strip().replace("", pd.NA).dropna().value_counts()
    )

    return {
        "total_candidates": int(len(df)),
        "downloaded_ok": int(df["download_ok"].fillna(False).sum()),
        "hard_prefilter_rejected": int((~df["hard_prefilter_keep"]).sum()),
        "ml_candidates": int(df["hard_prefilter_keep"].fillna(False).sum()),
        "final_kept": int(df["final_keep"].fillna(False).sum()),
        "top_reject_reasons": exploded.head(10).to_dict(),
    }


def save_positive_images(df: pd.DataFrame, output_dir: Path, keep_col: str = "final_keep") -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_df = df[df[keep_col].fillna(False)].copy()
    copied = 0

    for _, row in kept_df.iterrows():
        src = Path(row["local_path"])
        if not src.exists():
            continue
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied += 1

    return copied


def run_pipeline_for_url(
    url: str,
    output_dir: str,
    raw_dir: str | None = None,
    model_path: str = "models/best_model.pkl",
) -> Dict:
    output_root = Path(output_dir)
    raw_root = Path(raw_dir) if raw_dir else Path("data/raw")
    page_id = make_page_id(url)

    page_output_dir = output_root / page_id
    page_output_dir.mkdir(parents=True, exist_ok=True)
    final_keep_dir = page_output_dir / "final_keep"
    final_keep_dir.mkdir(parents=True, exist_ok=True)

    page_info = write_page_info(
        page_url=url,
        page_id=page_id,
        model_path=model_path,
        raw_dir=raw_root,
        output_dir=page_output_dir,
    )

    candidates_csv_path = page_output_dir / "candidates.csv"
    final_kept_csv_path = page_output_dir / "final_kept.csv"
    run_log_path = page_output_dir / "run_log.json"

    tmp_df = prepare_candidates_dataframe(url=url, raw_dir=raw_root)

    if tmp_df.empty:
        empty_candidates = pd.DataFrame(columns=FINAL_KEEP_COLUMNS)
        empty_candidates.to_csv(candidates_csv_path, index=False)
        empty_candidates.to_csv(final_kept_csv_path, index=False)

        summary = summarize_pipeline_results(tmp_df)
        summary.update(
            {
                "page_url": url,
                "page_id": page_id,
                "saved_images": 0,
                "top_reject_reasons": {},
                "paths_to_saved_artifacts": {
                    "page_info": str(page_output_dir / "page_info.json"),
                    "candidates_csv": str(candidates_csv_path),
                    "final_kept_csv": str(final_kept_csv_path),
                    "run_log": str(run_log_path),
                    "final_keep_dir": str(final_keep_dir),
                },
            }
        )
        run_log_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary.update(page_info)
        return summary

    df = download_candidates(tmp_df, raw_root)
    df = enrich_with_image_metadata(df)
    df = apply_hard_prefilter(df)
    df = apply_ml_filter(df=df, model_path=model_path)

    df = ensure_columns(df, FINAL_KEEP_COLUMNS)
    df.to_csv(candidates_csv_path, index=False)

    final_kept_df = df[df["final_keep"].fillna(False)].copy()
    final_kept_df = ensure_columns(final_kept_df, FINAL_KEEP_COLUMNS)
    final_kept_df[FINAL_KEEP_COLUMNS].to_csv(final_kept_csv_path, index=False)

    saved_images = save_positive_images(df, final_keep_dir, keep_col="final_keep")

    summary = summarize_pipeline_results(df)
    summary.update(
        {
            "page_url": url,
            "page_id": page_id,
            "saved_images": saved_images,
            "paths_to_saved_artifacts": {
                "page_info": str(page_output_dir / "page_info.json"),
                "candidates_csv": str(candidates_csv_path),
                "final_kept_csv": str(final_kept_csv_path),
                "run_log": str(run_log_path),
                "final_keep_dir": str(final_keep_dir),
            },
        }
    )

    run_log_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary.update(page_info)
    return summary