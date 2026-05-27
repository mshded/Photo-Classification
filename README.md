# Photo-Classification

## Постановка задачи
Проект фильтрует изображения страницы на два класса:
- `content`: фото, изображения/карточки товаров, обложки и содержательные иллюстрации.
- `non_content`: иконки, логотипы, кнопки, декоративные UI-элементы, баннеры, tracking/technical изображения.

Классификация выполняется для **целого изображения**. Основная метрика: **precision** (вместе с recall/F1/accuracy и confusion matrix).

## Архитектура
`URL -> parser -> candidates -> download -> metadata/features -> hard prefilter -> ML classifier -> deduplication -> final_keep/`

## Структура репозитория
- `notebooks/01_dataset_collection.ipynb` — сбор/разметка (без добавления новых данных на финальном этапе).
- `notebooks/02_eda.ipynb` — EDA и анализ распределений.
- `notebooks/03_training.ipynb` — **единственный** сценарий обучения/оценки/сохранения артефактов.
- `run_demo.py` — end-to-end demo уже обученной модели.
- `src/` — parser/features/classifier/pipeline/metrics.
- `data/labels.csv` — фиксированный датасет.
- `models/best_model.pkl` — артефакт модели.
- `results/*.csv` — артефакты оценки (main + stress-test).

## Датасет
Финальный датасет (без добавления новых данных):
- 650 изображений;
- 362 `content`;
- 288 `non_content`;
- 9 страниц.

## Evaluation protocol
1. **Основной**: `duplicate_safe_group_split(content_hash->canonical_image_url->normalized_image_url)`.
   - Предотвращает утечку дубликатов между train/val/test.
2. **Дополнительный stress-test**: `page_holdout_stress_test(page_stub->page_id->page_url)`.
   - Показывает перенос на новые страницы, но на 9 страницах нестабилен и не заменяет основную метрику.

## Модель
- `LogisticRegression` на metadata/URL features.
- `hard prefilter` перед ML для явного технического мусора.
- Подбор threshold только на validation.
- Финальная дедупликация в demo pipeline.

## Артефакты обучения (из `notebooks/03_training.ipynb`)
- `models/best_model.pkl`
- `results/metrics.csv`
- `results/threshold_metrics.csv`
- `results/split_assignment.csv`

## Запуск
```bash
pip install -r requirements.txt
# Открыть и полностью выполнить notebooks/03_training.ipynb
python run_demo.py --url "https://example.com" --model_path models/best_model.pkl
```
