from __future__ import annotations

import argparse

from src.classifier import train_and_save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML model and save artifacts")
    parser.add_argument(
        "--labels_csv",
        default="data/labels.csv",
        help="Path to labeled dataset CSV (default: data/labels.csv)",
    )
    parser.add_argument(
        "--model_path",
        default="models/best_model.pkl",
        help="Path for output model artifacts (default: models/best_model.pkl)",
    )
    parser.add_argument(
        "--model_type",
        default="logreg",
        choices=["logreg"],
        help="ML model type (default: logreg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_and_save_model(
        labels_csv_path=args.labels_csv,
        model_path=args.model_path,
        model_type=args.model_type,
    )

    print("Обучение завершено.")
    print(f"Порог: {report['threshold']:.3f}")
    print(f"Метрики train: {report['train_metrics']}")
    print(f"Метрики val: {report['val_metrics']}")
    print(f"Метрики test: {report['test_metrics']}")


if __name__ == "__main__":
    main()
