# AdventureWorks CRM Analytics Platform

This project is a small CRM analytics platform for AdventureWorks sales data.

It has one application package:

```text
app/
```

The platform includes:

- FastAPI API with Swagger docs
- Gradio dashboard
- Tableau dashboard embed
- EDA summaries and charts
- Buy / Not Buy prediction
- Sales amount prediction
- Future monthly sales forecasting

## Project Structure

```text
app/
├── api/
│   ├── routers/      # FastAPI endpoints
│   ├── schemas/      # Pydantic request/response models
│   └── services/     # Data, EDA, and business logic
├── dashboard/        # Single Gradio dashboard
├── ml/               # Classifier, regressor, forecasting, training
├── models/           # Package marker only
├── tests/            # Pytest tests
├── core.py           # Project paths
└── main.py           # FastAPI app
```

Data and generated artifacts stay outside the app:

```text
data/
models/
outputs/
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate EDA Data and Charts

```bash
python3 -m app.api.services.eda_pipeline
```

This creates:

- `data/processed/merged_sales_data.csv`
- `outputs/eda/summary.json`
- EDA chart images in `outputs/eda/`

## Train Models

```bash
python3 -m app.ml.training
```

This creates:

- `models/random_forest_classifier.joblib`
- `models/random_forest_regressor.joblib`

## Run FastAPI

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

Important API endpoints:

- `GET /api/eda/monthly-sales`
- `GET /api/eda/region-sales`
- `GET /api/eda/category-sales`
- `POST /api/predict/buy`
- `POST /api/predict/sales`
- `GET /api/forecast/monthly-sales`

## Run Gradio Dashboard

```bash
python3 -m app.dashboard.gradio_app
```

Open:

```text
http://127.0.0.1:7860
```

Dashboard tabs:

- Tableau Dashboard
- EDA Summary
- Interactive Analysis
- Buy / Not Buy Prediction
- Sales Prediction
- Future Sales Forecast

## Run FastAPI and Gradio Together

```bash
sh scripts/start.sh
```

FastAPI runs on port `8000`.
Gradio runs on port `7860`.

## Docker

```bash
docker build -t adventureworks-crm .
docker run -p 8000:8000 -p 7860:7860 adventureworks-crm
```

## Tests

```bash
pytest -q
```

## Short Workflow

```bash
source .venv/bin/activate
python3 -m app.api.services.eda_pipeline
python3 -m app.ml.training
pytest -q
sh scripts/start.sh
```
