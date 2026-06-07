# CRM проект AdventureWorks

Это учебный CRM/Data Science проект по данным продаж AdventureWorks.

Теперь в проекте есть одна главная папка приложения:

```text
app/
```

Внутри нее находятся:

- FastAPI API;
- один Gradio dashboard;
- Tableau dashboard embed;
- EDA анализ;
- Classification: `Buy / Not Buy`;
- Regression: прогноз суммы продаж;
- Forecasting: прогноз будущих продаж.

## Структура

```text
app/
├── api/
│   ├── routers/      # FastAPI endpoints
│   ├── schemas/      # Pydantic схемы
│   └── services/     # работа с данными и EDA
├── dashboard/        # один Gradio dashboard
├── ml/               # модели и обучение
├── tests/            # тесты
├── core.py           # пути проекта
└── main.py           # FastAPI app
```

Данные и результаты лежат отдельно:

```text
data/
models/
outputs/
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Шаг 1. Создать EDA и очищенные данные

```bash
python3 -m app.api.services.eda_pipeline
```

После этого появятся:

- `data/processed/merged_sales_data.csv`;
- `outputs/eda/summary.json`;
- графики в `outputs/eda/`.

## Шаг 2. Обучить модели

```bash
python3 -m app.ml.training
```

После этого появятся:

- `models/random_forest_classifier.joblib`;
- `models/random_forest_regressor.joblib`.

## Шаг 3. Запустить приложение локально

```bash
uvicorn app.main:app --reload --port 8000
```

Открыть:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

Gradio dashboard находится на `/`.

Главные endpoints:

- `GET /api/eda/monthly-sales`
- `GET /api/eda/region-sales`
- `GET /api/eda/category-sales`
- `POST /api/predict/buy`
- `POST /api/predict/sales`
- `GET /api/forecast/monthly-sales`

В dashboard есть вкладки:

- Tableau Dashboard;
- EDA Summary;
- Interactive Analysis;
- Buy / Not Buy Prediction;
- Sales Prediction;
- Future Sales Forecast.

## Запуск через startup script

```bash
sh scripts/start.sh
```

Скрипт использует переменную `PORT`, если она есть. Иначе использует `8000`.

## Docker

```bash
docker build -t adventureworks-crm .
docker run --rm -p 8000:8000 -e PORT=8000 adventureworks-crm
```

## Railway

1. Загрузи проект в GitHub.
2. В Railway создай новый проект из GitHub repo.
3. Railway найдет `Dockerfile` в корне проекта.
4. Запусти deploy.
5. Открой Railway domain.

Railway автоматически дает переменную `PORT`, а `scripts/start.sh` запускает приложение на этом порту.

## Тесты

```bash
pytest -q
```

## Короткий workflow

```bash
source .venv/bin/activate
python3 -m app.api.services.eda_pipeline
python3 -m app.ml.training
pytest -q
sh scripts/start.sh
```

## Что сказать на презентации

1. Я взял данные AdventureWorks.
2. Я очистил данные: удалил пропуски, дубликаты и выбросы.
3. Я сделал EDA: продажи по времени, регионам, товарам и клиентам.
4. Я построил classification модель для `Buy / Not Buy`.
5. Я построил regression модель для суммы продаж.
6. Я добавил прогноз будущих продаж.
7. Я сделал FastAPI API, Swagger docs и Gradio dashboard с Tableau.
