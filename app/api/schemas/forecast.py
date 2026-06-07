from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    month: str
    predicted_sales_amount: float


class MonthlySalesForecastResponse(BaseModel):
    product: Optional[str] = None
    region: Optional[str] = None
    horizon: int = Field(..., ge=1)
    predicted_sales_next_month: float
    predicted_sales_next_quarter: float
    forecast: list[ForecastPoint]
