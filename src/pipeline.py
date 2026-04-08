from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd

from src.classifier import load_model_artifacts, predict_proba
from src.features import (
    REPEATED_URL_THRESHOLD,
    extract_url_flags,
    has_extreme_aspect_ratio,
    is_probable_tracking_pixel,
    is_too_small,
)
from src.image_utils import download_image, get_image_metadata, make_unique_filename
from src.parser import collect_image_candidates


def _make_page_id(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.strip("/").replace("/", "_") or "home"
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)[:50]
    host = parsed.netloc.replace(":", "_")
    return f"{host}_{slug}".strip("_")


def prepare_candidates_dataframe(url: str, raw_dir: Path) -> pd.DataFrame:
    df = collect_image_candidates(url)
    if df.empty:
        return df

    page_id = _make_page_id(url)
    raw_page_dir = raw_dir / page_id

    df = df.copy()
    df["page_id"] = page_id
    df["raw_page_dir"] = str(raw_page_dir)

    # Make IDs stable and include page prefix for easier merges with labels
    df["candidate_id"] = [f"{page_id}_cand_{i:06d}" for i in range(len(df))]
    df["local_file_name"] = df.apply(
        lambda row: make_unique_filename(row["image_url"], row["candidate_id"]), axis=1
    )
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


def apply_baseline_rules(df: pd.DataFrame) -> pd.DataFrame:
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

        url_flags = extract_url_flags(
            row.get("image_url", ""),
            row.get("file_name", ""),
            row.get("alt_text", ""),
        )
        if url_flags["has_suspicious_keyword"]:
            flags.append("suspicious_keyword")
        if url_flags["has_tracking_hint"]:
            flags.append("tracking_url_hint")

        if is_probable_tracking_pixel(
            row.get("width"),
            row.get("height"),
            row.get("file_size_bytes"),
            row.get("image_url", ""),
            row.get("domain", ""),
        ):
            flags.append("probable_tracking_pixel")

        if is_too_small(row.get("width"), row.get("height"), row.get("area")):
            flags.append("too_small")

        extreme_ar = has_extreme_aspect_ratio(row.get("aspect_ratio"))
        if extreme_ar:
            flags.append("extreme_aspect_ratio")

        image_url = row.get("image_url", "")
        is_repeated = repeat_counts.get(image_url, 0) >= REPEATED_URL_THRESHOLD
        if is_repeated:
            flags.append("repeated_url")

        hard_reject = {
            "download_failed",
            "invalid_image",
            "probable_tracking_pixel",
            "too_small",
            "tracking_url_hint",
        }
        soft_reject = {
            "suspicious_keyword",
            "extreme_aspect_ratio",
            "repeated_url",
        }

        has_hard = any(f in hard_reject for f in flags)
        soft_count = sum(f in soft_reject for f in flags)
        keep = not has_hard and soft_count == 0

        keeps.append(keep)
        reasons.append("" if keep else ";".join(flags))
        flags_col.append(json.dumps(flags, ensure_ascii=False))

    out["baseline_keep"] = keeps
    out["baseline_reject_reason"] = reasons
    out["baseline_rule_flags"] = flags_col
    return out


def apply_ml_filter(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    out = df.copy()
    out["ml_score"] = pd.NA
    out["ml_pred"] = 0

    ml_candidates_mask = out["baseline_keep"].fillna(False)
    if not ml_candidates_mask.any():
        out["final_keep"] = out["baseline_keep"].fillna(False)
        return out

    artifacts = load_model_artifacts(model_path)
    model = artifacts["model"]
    threshold = float(artifacts.get("threshold", 0.5))

    ml_df = out.loc[ml_candidates_mask].copy()
    scores = predict_proba(model, ml_df)
    preds = (scores >= threshold).astype(int)

    out.loc[ml_candidates_mask, "ml_score"] = scores.values
    out.loc[ml_candidates_mask, "ml_pred"] = preds.values
    out["final_keep"] = out["baseline_keep"].fillna(False) & (out["ml_pred"] == 1)
    return out


def summarize_baseline_results(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "total_candidates": 0,
            "downloaded_ok": 0,
            "baseline_rejected": 0,
            "baseline_kept": 0,
            "ml_candidates": 0,
            "final_kept": 0,
            "top_reject_reasons": {},
        }

    rejected = df.loc[~df["baseline_keep"], "baseline_reject_reason"].fillna("")
    exploded = (
        rejected.str.split(";").explode().str.strip().replace("", pd.NA).dropna().value_counts()
    )

    return {
        "total_candidates": int(len(df)),
        "downloaded_ok": int(df["download_ok"].fillna(False).sum()),
        "baseline_rejected": int((~df["baseline_keep"]).sum()),
        "baseline_kept": int(df["baseline_keep"].sum()),
        "ml_candidates": int(df["baseline_keep"].fillna(False).sum()),
        "final_kept": int(df.get("final_keep", df["baseline_keep"]).fillna(False).sum()),
        "top_reject_reasons": exploded.head(10).to_dict(),
    }


def save_positive_images(df: pd.DataFrame, output_dir: Path, keep_col: str = "baseline_keep") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_df = df[df[keep_col].fillna(False)].copy()

    for _, row in kept_df.iterrows():
        src = Path(row["local_path"])
        if not src.exists():
            continue
        dst = output_dir / src.name
        shutil.copy2(src, dst)


def run_pipeline_for_url(
    url: str,
    output_dir: str,
    raw_dir: str | None = None,
    mode: str = "baseline_only",
    model_path: str = "models/best_model.pkl",
) -> Dict:
    output_root = Path(output_dir)
    raw_root = Path(raw_dir) if raw_dir else Path("data/raw")

    tmp_df = prepare_candidates_dataframe(url=url, raw_dir=raw_root)
    if tmp_df.empty:
        page_id = _make_page_id(url)
        page_output_dir = output_root / page_id
        page_output_dir.mkdir(parents=True, exist_ok=True)
        results_file = "baseline_results.csv" if mode == "baseline_only" else "ml_results.csv"
        (page_output_dir / results_file).write_text("", encoding="utf-8")
        summary = summarize_baseline_results(tmp_df)
        summary.update({"url": url, "page_id": page_id, "results_csv": str(page_output_dir / results_file)})
        return summary

    page_id = tmp_df["page_id"].iloc[0]
    page_output_dir = output_root / page_id
    page_output_dir.mkdir(parents=True, exist_ok=True)

    df = download_candidates(tmp_df, raw_root)
    df = enrich_with_image_metadata(df)
    df = apply_baseline_rules(df)

    if mode == "baseline_plus_ml":
        df = apply_ml_filter(df=df, model_path=model_path)
        results_csv = page_output_dir / "ml_results.csv"
        df.to_csv(results_csv, index=False)
        save_positive_images(df, page_output_dir, keep_col="final_keep")
    else:
        df["final_keep"] = df["baseline_keep"]
        results_csv = page_output_dir / "baseline_results.csv"
        df.to_csv(results_csv, index=False)
        save_positive_images(df, page_output_dir, keep_col="baseline_keep")

    summary = summarize_baseline_results(df)
    summary.update({"url": url, "page_id": page_id, "results_csv": str(results_csv), "mode": mode})
    return summary
