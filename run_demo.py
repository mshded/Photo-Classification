from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_pipeline_for_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6/7 demo: baseline and baseline+ML")
    parser.add_argument("--url", required=True, help="Page URL to process")
    parser.add_argument(
        "--output_dir",
        default="results/examples",
        help="Directory for outputs (default: results/examples)",
    )
    parser.add_argument(
        "--raw_dir",
        default="data/raw",
        help="Directory for raw downloaded candidates (default: data/raw)",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "ml"],
        default="baseline",
        help="baseline: stage 6 only, ml: baseline prefilter + ML",
    )
    parser.add_argument(
        "--model_path",
        default="models/best_model.pkl",
        help="Path to model artifacts for ml mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_mode = "baseline_only" if args.mode == "baseline" else "baseline_plus_ml"

    summary = run_pipeline_for_url(
        url=args.url,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        mode=pipeline_mode,
        model_path=args.model_path,
    )

    print(f"URL страницы: {summary.get('url', args.url)}")
    print(f"Режим: {args.mode}")
    print(f"Найдено кандидатов: {summary.get('total_candidates', 0)}")
    print(f"Успешно скачано изображений: {summary.get('downloaded_ok', 0)}")
    print(f"Отброшено baseline: {summary.get('baseline_rejected', 0)}")

    if args.mode == "ml":
        print(f"Передано в ML: {summary.get('ml_candidates', 0)}")
        print(f"Сохранено финально: {summary.get('final_kept', 0)}")
    else:
        print(f"Сохранено baseline-положительных: {summary.get('baseline_kept', 0)}")

    print(f"CSV результатов: {summary.get('results_csv', '')}")

    top_reasons = summary.get("top_reject_reasons", {})
    print("Топ причин отбрасывания:")
    if not top_reasons:
        print("  - нет")
    else:
        for reason, cnt in top_reasons.items():
            print(f"  - {reason}: {cnt}")

    results_csv = Path(summary.get("results_csv", ""))
    if results_csv.exists() and results_csv.stat().st_size == 0:
        print("Примечание: кандидаты на странице не найдены.")


if __name__ == "__main__":
    main()
