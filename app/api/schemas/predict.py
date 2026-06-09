from __future__ import annotations

from pydantic import BaseModel, Field


class BuyPredictionRequest(BaseModel):
    customer_key: int = Field(..., ge=0)
    product_key: int = Field(..., ge=0)
    list_price: float = Field(..., ge=0)
    country_region: str = Field(..., min_length=1)
    state_province: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    subcategory: str = Field(..., min_length=1)
    color: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)


class BuyPredictionResponse(BaseModel):
    prediction: str
    buy_probability: float


class SalesPredictionRequest(BuyPredictionRequest):
    order_quantity: float = Field(..., ge=0)
    unit_price: float = Field(..., ge=0)
    unit_price_discount_pct: float = Field(..., ge=0, le=1)
    month: str = Field(..., min_length=1)
    fiscal_quarter: str = Field(..., min_length=1)


class SalesPredictionResponse(BaseModel):
    predicted_sales_amount: float
