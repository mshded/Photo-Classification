from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_pipeline_for_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 rule-based baseline demo")
    parser.add_argument("--url", required=True, help="Page URL to process")
    parser.add_argument(
        "--output_dir",
        default="results/examples",
        help="Directory for baseline outputs (default: results/examples)",
    )
    parser.add_argument(
        "--raw_dir",
        default="data/raw",
        help="Directory for raw downloaded candidates (default: data/raw)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline_for_url(
        url=args.url,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
    )

    print(f"URL страницы: {summary.get('url', args.url)}")
    print(f"Найдено кандидатов: {summary.get('total_candidates', 0)}")
    print(f"Успешно скачано изображений: {summary.get('downloaded_ok', 0)}")
    print(f"Отброшено baseline: {summary.get('baseline_rejected', 0)}")
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
