from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.eda import CategorySalesItem, MonthlySalesItem, RegionSalesItem, SummaryResponse
from app.api.services import data_service

router = APIRouter()


@router.get("/summary", response_model=SummaryResponse)
async def get_summary() -> dict:
    return data_service.load_eda_summary()


@router.get("/monthly-sales", response_model=list[MonthlySalesItem])
async def get_monthly_sales() -> list[dict]:
    return data_service.monthly_sales()


@router.get("/region-sales", response_model=list[RegionSalesItem])
async def get_region_sales() -> list[dict]:
    return data_service.grouped_sales("region")


@router.get("/category-sales", response_model=list[CategorySalesItem])
async def get_category_sales() -> list[dict]:
    return data_service.grouped_sales("category")
