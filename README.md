# Photo-Classification

Учебный MVP для фильтрации изображений веб-страницы: система извлекает все candidate-картинки и сохраняет только **content**-изображения.

## Постановка задачи
- `content`: фотографии, карточки/изображения товаров, обложки, содержательные иллюстрации.
- `non_content`: иконки, логотипы, кнопки, decorative/UI-элементы, рекламные и tracking-изображения.
- Классификация выполняется для **целого изображения** (без segmentation/detection).
- Основная метрика: **precision** (всегда вместе с recall/F1 и количеством сохранённых изображений).

## Архитектура
`URL -> parser -> candidates -> download -> metadata/features -> hard prefilter -> ML classifier -> deduplication -> final_keep/`

## Структура
- `run_demo.py` — demo CLI.
- `train_model.py` — воспроизводимое обучение.
- `src/` — парсер, признаки, классификатор, pipeline, метрики.
- `data/labels.csv` — существующий датасет (без расширения).
- `models/best_model.pkl` — сохранённая модель.
- `results/metrics.csv`, `results/threshold_metrics.csv`, `results/split_assignment.csv` — артефакты обучения.
- `results/examples/<page_id>/` — артефакты demo.
- `tests/test_smoke.py` — smoke-тесты.

## Модель и воспроизводимость
- Модель: `LogisticRegression`.
- Split: `page_level_group_split(page_stub->page_id->page_url)`.
- Подбор threshold: только на `val`, с приоритетом precision.
- Проверка утечки страниц между `train/val/test` встроена (ошибка при пересечении групп).

## Запуск
```bash
pip install -r requirements.txt
python train_model.py
python run_demo.py --url "https://example.com" --model_path models/best_model.pkl
```

## Ограничения
- Небольшой фиксированный датасет и ограниченное число страниц.
- Baseline на metadata/URL features может деградировать на новых типах сайтов.
- Полная поддержка сложных SPA/infinite scroll не обязательна.
- В текущей среде без установки зависимостей (`pandas`, `sklearn`) обучение/тесты не выполнятся.
