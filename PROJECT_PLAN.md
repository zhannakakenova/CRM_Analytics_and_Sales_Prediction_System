# Project Plan

## Goal

Build one MVC-style CRM analytics platform with:

- data cleaning and EDA,
- FastAPI API with Swagger and ReDoc,
- Gradio dashboard,
- Tableau dashboard embed,
- RandomForestClassifier for `Buy / Not Buy`,
- RandomForestRegressor for sales amount prediction,
- monthly sales forecasting,
- pytest coverage,
- Docker startup.

## Main Application

The single source of truth is `app/`.

```text
app/
├── api/
│   ├── routers/
│   ├── schemas/
│   └── services/
├── dashboard/
├── ml/
│   ├── classifier/
│   ├── regressor/
│   ├── forecasting/
│   └── training.py
├── tests/
├── core.py
└── main.py
```

## Commands

Generate EDA outputs:

```bash
python3 -m app.api.services.eda_pipeline
```

Train models:

```bash
python3 -m app.ml.training
```

Run FastAPI:

```bash
uvicorn app.main:app --reload --port 8000
```

Run Gradio:

```bash
python3 -m app.dashboard.gradio_app
```

Run both:

```bash
sh scripts/start.sh
```

Run tests:

```bash
pytest -q
```

## Expected Outputs

- `data/processed/merged_sales_data.csv`
- `outputs/eda/*.png`
- `outputs/eda/summary.json`
- `models/random_forest_classifier.joblib`
- `models/random_forest_regressor.joblib`
