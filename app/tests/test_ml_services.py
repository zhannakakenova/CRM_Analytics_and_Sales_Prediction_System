from __future__ import annotations

from app.ml.classifier.service import CLASSIFIER_FEATURES
from app.ml.regressor import service as regressor_service


class FakeRegressor:
    def predict(self, frame):
        assert list(frame.columns) == regressor_service.REGRESSOR_FEATURES
        return [321.45]


def test_classifier_does_not_use_order_quantity() -> None:
    assert "order_quantity" not in CLASSIFIER_FEATURES


def test_sales_prediction_uses_trained_regressor(monkeypatch) -> None:
    monkeypatch.setattr(regressor_service, "load_regressor", lambda: FakeRegressor())

    prediction = regressor_service.predict_sales(
        {
            "customer_key": 11000,
            "product_key": 214,
            "order_quantity": 1,
            "unit_price": 34.99,
            "unit_price_discount_pct": 0.0,
            "list_price": 34.99,
            "country_region": "Australia",
            "state_province": "Queensland",
            "category": "Accessories",
            "subcategory": "Helmets",
            "color": "Red",
            "channel": "Internet",
            "region": "Australia",
            "month": "2017 Aug",
            "fiscal_quarter": "FY2018 Q1",
        }
    )

    assert prediction == 321.45
