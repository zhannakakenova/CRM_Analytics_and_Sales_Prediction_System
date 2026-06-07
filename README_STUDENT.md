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

## Run App Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

The Gradio dashboard is mounted at `/`.

Important API endpoints:

- `GET /api/eda/monthly-sales`
- `GET /api/eda/region-sales`
- `GET /api/eda/category-sales`
- `POST /api/predict/buy`
- `POST /api/predict/sales`
- `GET /api/forecast/monthly-sales`

Dashboard tabs:

- Tableau Dashboard
- EDA Summary
- Interactive Analysis
- Buy / Not Buy Prediction
- Sales Prediction
- Future Sales Forecast

## Run With Startup Script

```bash
sh scripts/start.sh
```

The startup script uses `PORT` if it exists, otherwise it uses `8000`.

## Docker

```bash
docker build -t adventureworks-crm .
docker run --rm -p 8000:8000 -e PORT=8000 adventureworks-crm
```

## Railway

1. Push the project to GitHub.
2. Create a new Railway project from the GitHub repo.
3. Railway will detect the root `Dockerfile`.
4. Deploy the service.
5. Open the Railway domain.

Railway provides the `PORT` variable automatically, and `scripts/start.sh` binds to it.

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
