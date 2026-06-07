from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.api.services.data_service import load_sales_data


def _monthly_series(product: Optional[str] = None, region: Optional[str] = None) -> pd.Series:
    df = load_sales_data()
    filtered = df.copy()

    if product:
        filtered = filtered[filtered["product"].astype(str) == product]
    if region:
        filtered = filtered[filtered["region"].astype(str) == region]
    if filtered.empty:
        filtered = df.copy()

    monthly = (
        filtered.dropna(subset=["order_date"])
        .set_index("order_date")
        .resample("ME")["sales_amount"]
        .sum()
        .asfreq("ME", fill_value=0)
    )
    return monthly


def _lag_frame(series: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"sales_amount": series})
    for lag in range(1, 4):
        data[f"lag_{lag}"] = data["sales_amount"].shift(lag)
    data["month_number"] = data.index.month
    data["trend"] = range(len(data))
    return data.dropna()


def forecast_monthly_sales(
    product: Optional[str] = None,
    region: Optional[str] = None,
    horizon: int = 6,
) -> dict:
    horizon = max(1, min(int(horizon), 24))
    series = _monthly_series(product=product, region=region)

    if len(series) < 6:
        baseline = float(series.mean()) if len(series) else 0.0
        last_date = pd.Timestamp.today().to_period("M").to_timestamp("M")
        points = [
            {
                "month": (last_date + pd.offsets.MonthEnd(step)).strftime("%Y-%m"),
                "predicted_sales_amount": baseline,
            }
            for step in range(1, horizon + 1)
        ]
    else:
        training = _lag_frame(series)
        features = ["lag_1", "lag_2", "lag_3", "month_number", "trend"]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(training[features], training["sales_amount"])

        history = series.astype(float).tolist()
        current_date = series.index.max()
        points = []
        for step in range(1, horizon + 1):
            next_date = current_date + pd.offsets.MonthEnd(step)
            row = pd.DataFrame(
                [
                    {
                        "lag_1": history[-1],
                        "lag_2": history[-2],
                        "lag_3": history[-3],
                        "month_number": next_date.month,
                        "trend": len(history),
                    }
                ]
            )
            prediction = max(0.0, float(model.predict(row)[0]))
            history.append(prediction)
            points.append(
                {
                    "month": next_date.strftime("%Y-%m"),
                    "predicted_sales_amount": prediction,
                }
            )

    next_month = points[0]["predicted_sales_amount"]
    next_quarter = sum(point["predicted_sales_amount"] for point in points[: min(3, len(points))])
    return {
        "product": product,
        "region": region,
        "horizon": horizon,
        "predicted_sales_next_month": next_month,
        "predicted_sales_next_quarter": next_quarter,
        "forecast": points,
    }
