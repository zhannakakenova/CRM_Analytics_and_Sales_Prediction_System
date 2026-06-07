from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from app.core import MODELS_DIR

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


@lru_cache(maxsize=1)
def load_regressor():
    return joblib.load(MODELS_DIR / "random_forest_regressor.joblib")


def predict_sales(input_row: dict) -> float:
    model = load_regressor()
    frame = pd.DataFrame([input_row])[REGRESSOR_FEATURES]
    return float(model.predict(frame)[0])
