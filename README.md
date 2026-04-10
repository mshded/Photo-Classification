# Photo Classification Project

Учебный MVP для фильтрации изображений со страниц сайтов с фокусом на **precision**:
оставляем только содержательные изображения страницы и отбрасываем иконки, кнопки,
декоративные/повторяющиеся UI-элементы, логотипы и рекламные баннеры.

## Что делает пайплайн

Для заданного URL пайплайн:
1. Собирает кандидаты изображений из обычного HTML (`src/parser.py`, `requests + BeautifulSoup`).
2. Скачивает кандидаты в `data/raw/<page_id>/`.
3. Извлекает метаданные изображений (валидность, размеры, формат, aspect ratio и т.д.).
4. Применяет baseline-фильтрацию (правила для отсечения явного мусора).
5. Применяет ML-фильтр к baseline-кандидатам.
6. Сохраняет финальные содержательные изображения и артефакты demo-запуска.

Проект ориентирован на обычные HTML-сайты и простую подгрузку (без усложнений типа SPA-first/segmentation/OCR).

## Установка

```bash
pip install -r requirements.txt
```

## Обучение модели

```bash
python run_train.py --labels_csv data/labels.csv --model_path models/best_model.pkl
```

## Запуск demo

```bash
python run_demo.py --url "https://example.com/page" --model_path models/best_model.pkl
```
Дополнительные аргументы run_demo:
- `--output_dir` (по умолчанию `results/examples`)
- `--raw_dir` (по умолчанию `data/raw`)

## Артефакты demo-запуска

Для каждого URL создаётся папка:

`results/examples/<page_id>/`

Внутри:
- `page_info.json` — параметры запуска страницы;
- `candidates.csv` — все кандидаты после скачивания/обогащения и фильтрации;
- `final_kept.csv` — только строки с `final_keep == True`;
- `run_log.json` — сводка запуска (counts, причины отбрасывания, пути к артефактам);
- `final_keep/` — реально сохранённые финальные изображения.
