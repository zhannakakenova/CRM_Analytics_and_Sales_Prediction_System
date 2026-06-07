from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from app.core import MODELS_DIR

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


@lru_cache(maxsize=1)
def load_classifier():
    return joblib.load(MODELS_DIR / "random_forest_classifier.joblib")


def predict_buy(input_row: dict) -> tuple[str, float]:
    model = load_classifier()
    frame = pd.DataFrame([input_row])[CLASSIFIER_FEATURES]
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return ("Buy" if prediction == 1 else "Not Buy"), probability
