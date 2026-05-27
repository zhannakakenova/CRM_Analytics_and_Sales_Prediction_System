# CRM School Project (Very Simple Guide)

This project helps you do 4 things:
1. Clean sales data
2. Explore data with charts
3. Train 2 machine learning models
4. Show everything in Gradio dashboard

## Project Files
- `src/data_exploration.py` -> data cleaning + charts
- `src/train_models.py` -> training models
- `src/predict_classification.py` -> Pipeline 1 (Buy / Not Buy)
- `src/predict_regression.py` -> Pipeline 2 (Sales Amount)
- `src/dashboard_gradio.py` -> dashboard app

## Step 0: Install
Open terminal in project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Run Data Exploration
This step cleans data and creates charts.

```bash
python3 src/data_exploration.py
```

You will get:
- `data/processed/merged_sales_data.csv`
- chart images in `outputs/eda/`

## Step 2: Run Training
This step trains both required models:
- `RandomForestClassifier` (Buy / Not Buy)
- `RandomForestRegressor` (predict sales amount)

```bash
python3 src/train_models.py
```

You will get model files in `models/`.

## Step 3A: Run Classification Pipeline (Buy / Not Buy)
What it means:
- Input: customer + product information
- Output: `Buy` or `Not Buy` + probability

Example:

```bash
python3 src/predict_classification.py \
  --customer_key 11000 \
  --product_key 214 \
  --order_quantity 1 \
  --list_price 1200 \
  --country_region "United States" \
  --state_province "California" \
  --category "Bikes" \
  --subcategory "Mountain Bikes" \
  --color "Black" \
  --channel "Reseller" \
  --region "Northwest"
```

## Step 3B: Run Regression Pipeline (Sales Amount)
What it means:
- Input: customer + product + pricing + time info
- Output: predicted sales number (`sales_amount`)

Example:

```bash
python3 src/predict_regression.py \
  --customer_key 11000 \
  --product_key 214 \
  --order_quantity 1 \
  --unit_price 1000 \
  --unit_price_discount_pct 0.0 \
  --list_price 1200 \
  --country_region "United States" \
  --state_province "California" \
  --category "Bikes" \
  --subcategory "Mountain Bikes" \
  --color "Black" \
  --channel "Reseller" \
  --region "Northwest" \
  --month "January" \
  --fiscal_quarter "Q1"
```

## Step 4: Open Gradio Dashboard

```bash
python3 src/dashboard_gradio.py
```

Then open the local link from terminal (usually `http://127.0.0.1:7860`).

Dashboard has:
- EDA summary and charts
- Interactive Data Analysis tab: filter current data and make new analysis tables/charts
- Classification tab: predicts `Buy` or `Not Buy`
- Regression tab: predicts `sales_amount`

## Easy Workflow (Short)
1. `python3 src/data_exploration.py`
2. `python3 src/train_models.py`
3. `python3 src/dashboard_gradio.py`

## If Something Fails
- Check you activated venv: `source .venv/bin/activate`
- Check models exist in `models/`
- Run EDA first, then training, then dashboard
