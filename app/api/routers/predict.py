from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.predict import (
    BuyPredictionRequest,
    BuyPredictionResponse,
    SalesPredictionRequest,
    SalesPredictionResponse,
)
from app.ml.classifier.service import predict_buy
from app.ml.regressor.service import predict_sales

router = APIRouter()


def _schema_to_dict(payload):
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


@router.post("/buy", response_model=BuyPredictionResponse)
async def predict_buy_endpoint(payload: BuyPredictionRequest) -> BuyPredictionResponse:
    prediction, probability = predict_buy(_schema_to_dict(payload))
    return BuyPredictionResponse(prediction=prediction, buy_probability=probability)


@router.post("/sales", response_model=SalesPredictionResponse)
async def predict_sales_endpoint(payload: SalesPredictionRequest) -> SalesPredictionResponse:
    prediction = predict_sales(_schema_to_dict(payload))
    return SalesPredictionResponse(predicted_sales_amount=prediction)
