# AdventureWorks Analytics Platform

This project is a small CRM analytics application built with Python. It uses AdventureWorks sales data to help students practice data preparation, exploratory data analysis (EDA), machine learning, forecasting, API development, and dashboard building.

web-production-03267.up.railway.app

The application has two main parts:

- A **FastAPI backend** with endpoints for analytics, predictions, and forecasting.
- A **Gradio dashboard** where users can explore charts and run model predictions from a browser.

## What You Will Learn

By studying this project, you will see how a real data application is organized:

- Load and merge several CSV tables into one analysis dataset.
- Clean data by removing duplicates, missing values, and outliers.
- Create EDA summaries and charts.
- Train machine learning models with scikit-learn.
- Save and load trained models with joblib.
- Build REST API endpoints with FastAPI.
- Create an interactive dashboard with Gradio.
- Test API behavior with pytest.
- Package and run the app with Docker.

## Project Structure

```text
jeanne-crm/
├── app/
│   ├── api/
│   │   ├── routers/          # FastAPI route definitions
│   │   ├── schemas/          # Request and response models
│   │   └── services/         # Data loading, cleaning, and EDA logic
│   ├── dashboard/            # Gradio dashboard UI
│   ├── ml/                   # Machine learning and forecasting code
│   ├── tests/                # API tests
│   ├── core.py               # Shared project paths
│   └── main.py               # Main FastAPI + Gradio app
├── data/
│   ├── csv/                  # Source CSV files
│   ├── processed/            # Clean merged dataset
│   └── database/             # SQLite database file
├── models/                   # Saved trained ML models
├── outputs/
│   └── eda/                  # Generated EDA charts and summary
├── scripts/
│   └── start.sh              # Production start command
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

## Main Features

### 1. EDA and Analytics

The EDA pipeline creates:

- A cleaned dataset: `data/processed/merged_sales_data.csv`
- Summary statistics: `outputs/eda/summary.json`
- Charts such as monthly sales, sales by region, top products, and top customers

Important file:

- `app/api/services/eda_pipeline.py`

### 2. Machine Learning Predictions

The project includes two Random Forest models:

- **Buy / Not Buy classifier**: predicts whether a customer is likely to buy a product.
- **Sales amount regressor**: predicts the expected sales amount.

The classifier creates synthetic `Not Buy` examples from customer-product pairs that do not appear in historical purchases. Order quantity is excluded from classification to prevent target leakage. The Sales Prediction service returns output from the trained Random Forest Regressor rather than directly calculating a sales formula.

Saved model files:

- `models/random_forest_classifier.joblib`
- `models/random_forest_regressor.joblib`

Important files:

- `app/ml/training.py`
- `app/ml/classifier/service.py`
- `app/ml/regressor/service.py`

### 3. Sales Forecasting

The forecasting service predicts future monthly sales. It can forecast overall sales or filter by product and region.

Important file:

- `app/ml/forecasting/service.py`

### 4. API Endpoints

FastAPI exposes the project features as HTTP endpoints.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/eda/summary` | Return EDA summary statistics |
| `GET` | `/api/eda/monthly-sales` | Return monthly sales totals |
| `GET` | `/api/eda/region-sales` | Return sales grouped by region |
| `GET` | `/api/eda/category-sales` | Return sales grouped by product category |
| `POST` | `/api/predict/buy` | Predict Buy / Not Buy |
| `POST` | `/api/predict/sales` | Predict sales amount |
| `GET` | `/api/forecast/monthly-sales` | Forecast future monthly sales |

FastAPI also provides automatic API documentation at:

- `http://localhost:8000/docs`

### 5. Gradio Dashboard

The dashboard is mounted at the root URL:

- `http://localhost:8000/`

Dashboard tabs include:

- Tableau Dashboard
- EDA Summary
- Interactive Analysis
- Buy / Not Buy Prediction
- Sales Prediction
- Future Sales Forecast

For a detailed explanation of every tab, its inputs, outputs, and suggested student workflow, read [`docs/DASHBOARD_GUIDE.md`](docs/DASHBOARD_GUIDE.md).

For details about classification, synthetic Not Buy records, regression, and forecasting, read [`docs/ML_METHODOLOGY.md`](docs/ML_METHODOLOGY.md).

Important file:

- `app/dashboard/gradio_app.py`

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Processed Data and EDA Outputs

If `data/processed/merged_sales_data.csv` or `outputs/eda/summary.json` is missing, run:

```bash
python3 -m app.api.services.eda_pipeline
```

### 4. Train the Models

If the files inside `models/` are missing, run:

```bash
python3 -m app.ml.training
```

This trains the classifier and regressor, then saves them as `.joblib` files.

### 5. Run the Application

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- Dashboard: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`

## Run with Docker

Build the Docker image:

```bash
docker build -t adventureworks-analytics .
```

Run the container:

```bash
docker run -p 8000:8000 adventureworks-analytics
```

Then open `http://localhost:8000/`.

## Run Tests

```bash
pytest
```

The tests check that the API endpoints respond correctly and that prediction payload validation works.

## Example API Requests

### Monthly Sales

```bash
curl http://localhost:8000/api/eda/monthly-sales
```

### Buy / Not Buy Prediction

```bash
curl -X POST http://localhost:8000/api/predict/buy \
  -H "Content-Type: application/json" \
  -d '{
    "customer_key": 11000,
    "product_key": 214,
    "list_price": 1200,
    "country_region": "United States",
    "state_province": "California",
    "category": "Bikes",
    "subcategory": "Mountain Bikes",
    "color": "Black",
    "channel": "Reseller",
    "region": "Northwest"
  }'
```

### Sales Prediction

```bash
curl -X POST http://localhost:8000/api/predict/sales \
  -H "Content-Type: application/json" \
  -d '{
    "customer_key": 11000,
    "product_key": 214,
    "order_quantity": 1,
    "unit_price": 1000,
    "unit_price_discount_pct": 0.0,
    "list_price": 1200,
    "country_region": "United States",
    "state_province": "California",
    "category": "Bikes",
    "subcategory": "Mountain Bikes",
    "color": "Black",
    "channel": "Reseller",
    "region": "Northwest",
    "month": "January",
    "fiscal_quarter": "Q1"
  }'
```

### Monthly Forecast

```bash
curl "http://localhost:8000/api/forecast/monthly-sales?horizon=6"
```

You can also filter by product and region:

```bash
curl "http://localhost:8000/api/forecast/monthly-sales?product=Mountain-200%20Black,%2046&region=Northwest&horizon=6"
```

## Recommended Student Workflow

If you are learning from this project, follow this order:

1. Start with `app/core.py` to understand where files are stored.
2. Read `app/api/services/dataset_service.py` to learn how raw tables are merged.
3. Run `python3 -m app.api.services.eda_pipeline` and inspect `outputs/eda/`.
4. Read `app/ml/training.py` to understand model training.
5. Start the app and explore the Gradio dashboard.
6. Open `http://localhost:8000/docs` and test the API endpoints.
7. Read `app/tests/test_api.py` to see how API behavior is tested.

## Notes for Students

- The source data is stored in `data/csv/`.
- The processed dataset is generated from the source CSV files.
- The dashboard depends on the processed dataset and saved model files.
- If something is missing, rerun the EDA pipeline and model training commands.
- API schemas in `app/api/schemas/` define which fields are required in requests.
- The project is intentionally modular, so each folder has a clear responsibility.

## Troubleshooting

### `FileNotFoundError: Processed data not found`

Run:

```bash
python3 -m app.api.services.eda_pipeline
```

### Model file is missing

Run:

```bash
python3 -m app.ml.training
```

### Model compatibility errors

The saved models require the pinned scikit-learn version from `requirements.txt`. Always launch the app through the active environment:

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

For Docker, rebuild the image after dependency changes:

```bash
docker build --no-cache -t adventureworks-analytics .
docker run -p 8000:8000 adventureworks-analytics
```

### Port 8000 is already in use

Run the app on another port:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

Then open `http://localhost:8001/`.

## Short Project Summary

This project turns AdventureWorks sales data into a complete analytics platform. It prepares data, creates visual summaries, trains machine learning models, exposes predictions through an API, and provides a browser dashboard for interactive exploration.
