from __future__ import annotations

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
    """Return buy prediction and probability using the classification model."""
    model = joblib.load(MODELS_DIR / "random_forest_classifier.joblib")
    X = pd.DataFrame([input_row])[CLASSIFIER_FEATURES]
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return pred, proba


def predict_sales(input_row: dict) -> float:
    """Return predicted sales amount using the regression model."""
    model = joblib.load(MODELS_DIR / "random_forest_regressor.joblib")
    X = pd.DataFrame([input_row])[REGRESSOR_FEATURES]
    return float(model.predict(X)[0])
