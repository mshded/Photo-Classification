# Photo Classification Project

Для заданного URL пайплайн:
1. Собирает кандидаты изображений через `src/parser.py`.
2. Скачивает кандидаты в `data/raw/<page_id>/` через `src/image_utils.py`.
3. Извлекает метаданные (валидность, размер, формат, площадь, aspect ratio).
4. Применяет прозрачные правила (small size, suspicious keywords, tracking-like мусор, extreme aspect ratio, repeated URL).
5. Сохраняет:
   - полный CSV с решениями baseline: `results/examples/<page_id>/baseline_results.csv`
   - только baseline-положительные изображения: `results/examples/<page_id>/`

## Запуск

```bash
python run_demo.py --url "https://example.com/page"
```

## Оценка качества

В `src/metrics.py` есть:
- `compute_classification_metrics(y_true, y_pred)`
- `evaluate_baseline_on_labels(labels_csv_path)`

