from __future__ import annotations

import argparse

from src.pipeline import run_pipeline_for_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo: image content filtering pipeline")
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
        "--model_path",
        default="models/best_model.pkl",
        help="Path to model artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = run_pipeline_for_url(
        url=args.url,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        model_path=args.model_path,
    )

    print(f"URL страницы: {summary.get('page_url', args.url)}")
    print(f"Page ID: {summary.get('page_id', '-')}")
    print(f"Найдено кандидатов: {summary.get('total_candidates', 0)}")
    print(f"Успешно скачано изображений: {summary.get('downloaded_ok', 0)}")
    print(f"Отброшено hard prefilter: {summary.get('hard_prefilter_rejected', 0)}")
    print(f"Передано в ML: {summary.get('ml_candidates', 0)}")
    print(f"Финально оставлено: {summary.get('final_kept', 0)}")
    print(f"Удалено финальной дедупликацией: {summary.get('final_dedup_removed', 0)}")

    if summary.get("total_candidates", 0) == 0:
        print("Сообщение: на странице не найдено кандидатов изображений.")
    elif summary.get("final_kept", 0) == 0:
        print("Сообщение: содержательные изображения не найдены после фильтрации.")

    top_reasons = summary.get("top_reject_reasons", {})
    print("Топ причин отбрасывания:")
    if not top_reasons:
        print("  - нет")
    else:
        for reason, cnt in top_reasons.items():
            print(f"  - {reason}: {cnt}")

    artifacts = summary.get("paths_to_saved_artifacts", {})
    print("Артефакты:")
    print(f"  page_info.json: {artifacts.get('page_info', '')}")
    print(f"  candidates.csv: {artifacts.get('candidates_csv', '')}")
    print(f"  final_kept.csv: {artifacts.get('final_kept_csv', '')}")
    print(f"  run_log.json: {artifacts.get('run_log', '')}")
    print(f"  final_keep/: {artifacts.get('final_keep_dir', '')}")


if __name__ == "__main__":
    main()