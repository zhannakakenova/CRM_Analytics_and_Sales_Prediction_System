from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")


CLASSIFIER_FEATURES = [
    "customer_key",
    "product_key",
    "order_quantity",
    "country_region",
    "state_province",
    "category",
    "subcategory",
    "color",
    "list_price",
    "channel",
    "region",
]

REGRESSOR_FEATURES = [
    "customer_key",
    "product_key",
    "order_quantity",
    "unit_price",
    "unit_price_discount_pct",
    "list_price",
    "country_region",
    "state_province",
    "category",
    "subcategory",
    "color",
    "channel",
    "region",
    "month",
    "fiscal_quarter",
]


def predict_buy(input_row: dict) -> tuple[int, float]:
    model = joblib.load(MODELS_DIR / "random_forest_classifier.joblib")
    X = pd.DataFrame([input_row])[CLASSIFIER_FEATURES]
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return pred, proba


def predict_sales(input_row: dict) -> float:
    model = joblib.load(MODELS_DIR / "random_forest_regressor.joblib")
    X = pd.DataFrame([input_row])[REGRESSOR_FEATURES]
    return float(model.predict(X)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prediction sample")
    parser.add_argument("--customer_key", type=int, required=True)
    parser.add_argument("--product_key", type=int, required=True)
    parser.add_argument("--order_quantity", type=float, default=1)
    parser.add_argument("--unit_price", type=float, default=0.0)
    parser.add_argument("--unit_price_discount_pct", type=float, default=0.0)
    parser.add_argument("--list_price", type=float, default=0.0)
    parser.add_argument("--country_region", type=str, default="United States")
    parser.add_argument("--state_province", type=str, default="California")
    parser.add_argument("--category", type=str, default="Bikes")
    parser.add_argument("--subcategory", type=str, default="Mountain Bikes")
    parser.add_argument("--color", type=str, default="Black")
    parser.add_argument("--channel", type=str, default="Reseller")
    parser.add_argument("--region", type=str, default="Northwest")
    parser.add_argument("--month", type=str, default="January")
    parser.add_argument("--fiscal_quarter", type=str, default="Q1")
    args = parser.parse_args()

    row = vars(args)

    buy_pred, buy_prob = predict_buy(row)
    sales_pred = predict_sales(row)

    label = "Buy" if buy_pred == 1 else "Not Buy"
    print(f"Classification: {label} (probability={buy_prob:.3f})")
    print(f"Regression (predicted sales_amount): {sales_pred:.2f}")


if __name__ == "__main__":
    main()
