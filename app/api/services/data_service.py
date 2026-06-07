from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from app.core import DATA_PATH, EDA_DIR


@lru_cache(maxsize=1)
def load_sales_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Processed data not found. Run python3 -m app.api.services.eda_pipeline first.")

    df = pd.read_csv(DATA_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


def load_eda_summary() -> dict:
    summary_path = EDA_DIR / "summary.json"
    if not summary_path.exists():
        return {}

    with open(summary_path, "r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def monthly_sales() -> list[dict]:
    df = load_sales_data()
    result = (
        df.dropna(subset=["order_date"])
        .set_index("order_date")
        .resample("ME")["sales_amount"]
        .sum()
        .reset_index()
    )
    result["month"] = result["order_date"].dt.strftime("%Y-%m")
    return result[["month", "sales_amount"]].to_dict(orient="records")


def grouped_sales(column: str) -> list[dict]:
    df = load_sales_data()
    result = (
        df.groupby(column, dropna=False)["sales_amount"]
        .sum()
        .reset_index()
        .sort_values("sales_amount", ascending=False)
    )
    result[column] = result[column].fillna("Unknown").astype(str)
    return result.to_dict(orient="records")
