from __future__ import annotations

from fastapi import FastAPI

from app.api.routers import eda, forecast, predict

app = FastAPI(
    title="AdventureWorks Analytics Platform",
    description="CRM analytics API with EDA, RandomForest predictions, and sales forecasting.",
    version="1.0.0",
)

app.include_router(eda.router, prefix="/api/eda", tags=["EDA"])
app.include_router(predict.router, prefix="/api/predict", tags=["Predictions"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecasting"])


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "AdventureWorks Analytics Platform",
        "docs": "/docs",
        "redoc": "/redoc",
    }
