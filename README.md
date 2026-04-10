# Photo Classification Project

Учебный **ML-based MVP** для фильтрации изображений со страниц сайтов с фокусом на **precision**:
оставляем только содержательные изображения страницы и отбрасываем иконки, кнопки,
декоративные/повторяющиеся UI-элементы, логотипы, рекламные баннеры и tracking-пиксели.

## Финальный пайплайн (один публичный сценарий)

Для заданного URL пайплайн:
1. Собирает кандидаты изображений из обычного HTML (`src/parser.py`, `requests + BeautifulSoup`).
2. Скачивает кандидаты в `data/raw/<page_id>/`.
3. Извлекает метаданные изображений (валидность, размеры, формат, aspect ratio и т.д.).
4. Применяет встроенный **hard prefilter** для явного технического мусора.
5. Для оставшихся кандидатов считает ML-score и применяет порог из сохранённых артефактов модели.
6. Сохраняет только `final_keep` изображения и артефакты demo-запуска.

> В проекте нет user-facing baseline режима: используется один ML pipeline
> с детерминированным hard prefilter до модели.

## Установка

```bash
pip install -r requirements.txt
```

## Обучение модели (канонический путь)

Обучение и анализ метрик выполняются в ноутбуке:

- `notebooks/03_training.ipynb`

Основная логика обучения/фичей/метрик остаётся в `src/classifier.py`, `src/features.py`, `src/metrics.py`.
`load_labeled_data()` по умолчанию пересобирает `train/val/test` через group-aware split
(content hash -> normalized image_url -> page_url -> fallback), чтобы старый `split` из CSV
не обходил честное разбиение.

### Как выбирается порог

- `train`: fit модели;
- `val`: подбор threshold под высокий precision (`select_threshold_for_precision`);
- `test`: финальная оценка с зафиксированным порогом.

Выбранный `threshold` сохраняется в `models/best_model.pkl` и используется в demo pipeline.

## Запуск demo

```bash
python run_demo.py --url "https://example.com/page" --model_path models/best_model.pkl
```

Дополнительные аргументы `run_demo.py`:
- `--output_dir` (по умолчанию `results/examples`)
- `--raw_dir` (по умолчанию `data/raw`)

## Артефакты demo-запуска

Для каждого URL создаётся папка:

`results/examples/<page_id>/`

Минимальный набор артефактов:
- `page_info.json` — параметры запуска;
- `candidates.csv` — кандидаты с тех. метаданными, hard prefilter и ML-результатами;
- `final_kept.csv` — только строки с `final_keep == True`;
- `run_log.json` — сводка запуска (counts, причины hard reject, пути);
- `final_keep/` — реально сохранённые финальные изображения.

Эти demo-артефакты генерируются локально для демонстрации и **не должны храниться в Git**
(`results/` и `models/` уже исключены через `.gitignore`).

Эти demo-артефакты генерируются локально для демонстрации и **не должны храниться в Git**
(`results/` и `models/` уже исключены через `.gitignore`).