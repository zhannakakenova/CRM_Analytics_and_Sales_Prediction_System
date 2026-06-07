from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class MonthlySalesItem(BaseModel):
    month: str
    sales_amount: float


class RegionSalesItem(BaseModel):
    region: str
    sales_amount: float


class CategorySalesItem(BaseModel):
    category: str
    sales_amount: float


class SummaryResponse(BaseModel):
    rows_after_cleaning: Optional[int] = None
    columns: Optional[int] = None
    date_range: Optional[list[str]] = None
    total_sales: Optional[float] = None
    average_sales: Optional[float] = None
    top_country_by_sales: Optional[list[str]] = None
