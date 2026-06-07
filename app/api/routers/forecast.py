from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.api.schemas.forecast import MonthlySalesForecastResponse
from app.ml.forecasting.service import forecast_monthly_sales

router = APIRouter()


@router.get("/monthly-sales", response_model=MonthlySalesForecastResponse)
async def get_monthly_sales_forecast(
    product: Optional[str] = Query(default=None),
    region: Optional[str] = Query(default=None),
    horizon: int = Query(default=6, ge=1, le=24),
) -> dict:
    return forecast_monthly_sales(product=product, region=region, horizon=horizon)
