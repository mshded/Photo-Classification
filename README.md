# Photo Classification Project

Чистая production-версия проекта для отбора контентных изображений с помощью ML-модели.

## Что делает пайплайн

Для заданного URL пайплайн:
1. Собирает кандидаты изображений через `src/parser.py`.
2. Скачивает кандидаты в `data/raw/<page_id>/` через `src/image_utils.py`.
3. Извлекает метаданные (валидность, размер, формат, площадь, aspect ratio).
4. Применяет prefilter-правила для явного мусора (tracking/small/invalid).
5. Применяет ML-модель (`LogisticRegression`) к оставшимся кандидатам.
6. Сохраняет результаты в `results/examples/<page_id>/baseline_results.csv` и изображения `final_keep`.

## Установка

```bash
pip install -r requirements.txt
```

## Обучение ML-модели

```bash
python run_train.py --labels_csv data/labels.csv --model_path models/best_model.pkl
```

## Запуск demo

```bash
python run_demo.py --url "https://example.com/page" --model_path models/best_model.pkl
```

python run_demo.py --url "https://eda.rambler.ru/media/recepty/recepty-kulichey-na-pashu-ot-klassiki-do-bystryh-variantov" --model_path models/best_model.pkl