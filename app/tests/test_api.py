from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


BUY_PAYLOAD = {
    "customer_key": 11000,
    "product_key": 214,
    "list_price": 1200,
    "country_region": "United States",
    "state_province": "California",
    "category": "Bikes",
    "subcategory": "Mountain Bikes",
    "color": "Black",
    "channel": "Reseller",
    "region": "Northwest",
}

SALES_PAYLOAD = {
    **BUY_PAYLOAD,
    "order_quantity": 1,
    "unit_price": 1000,
    "unit_price_discount_pct": 0.0,
    "month": "January",
    "fiscal_quarter": "Q1",
}


def test_monthly_sales_endpoint() -> None:
    response = client.get("/api/eda/monthly-sales")
    assert response.status_code == 200
    assert len(response.json()) > 0
    assert {"month", "sales_amount"} <= set(response.json()[0])


def test_region_sales_endpoint() -> None:
    response = client.get("/api/eda/region-sales")
    assert response.status_code == 200
    assert len(response.json()) > 0
    assert {"region", "sales_amount"} <= set(response.json()[0])


def test_category_sales_endpoint() -> None:
    response = client.get("/api/eda/category-sales")
    assert response.status_code == 200
    assert len(response.json()) > 0
    assert {"category", "sales_amount"} <= set(response.json()[0])


def test_classification_endpoint() -> None:
    response = client.post("/api/predict/buy", json=BUY_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Buy", "Not Buy"}
    assert 0 <= body["buy_probability"] <= 1


def test_regression_endpoint() -> None:
    response = client.post("/api/predict/sales", json=SALES_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["predicted_sales_amount"] >= 0


def test_forecast_endpoint() -> None:
    response = client.get("/api/forecast/monthly-sales?horizon=3")
    assert response.status_code == 200
    body = response.json()
    assert body["horizon"] == 3
    assert len(body["forecast"]) == 3


def test_invalid_prediction_payload() -> None:
    response = client.post("/api/predict/sales", json={"customer_key": -1})
    assert response.status_code == 422
