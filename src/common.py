from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

DATA_DIR = Path("data/csv")


def load_tables(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all source CSV tables."""
    return {
        "sales": pd.read_csv(data_dir / "sales_data.csv"),
        "orders": pd.read_csv(data_dir / "sales_order_data.csv"),
        "customers": pd.read_csv(data_dir / "customer_data.csv"),
        "products": pd.read_csv(data_dir / "product_data.csv"),
        "territories": pd.read_csv(data_dir / "sales_territory_data.csv"),
        "dates": pd.read_csv(data_dir / "date_data.csv"),
    }


def build_merged_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Merge all tables into one analysis-ready dataset."""
    t = load_tables(data_dir)

    df = t["sales"].merge(t["orders"], on="sales_order_line_key", how="left")
    df = df.merge(t["customers"], on="customer_key", how="left")
    df = df.merge(t["products"], on="product_key", how="left")
    df = df.merge(t["territories"], on="sales_territory_key", how="left")

    date_info = t["dates"][["date_key", "full_date", "month", "fiscal_quarter", "fiscal_year"]].copy()
    df = df.merge(date_info, left_on="order_date_key", right_on="date_key", how="left")
    df = df.rename(columns={"full_date": "order_date"})
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: drop duplicates, missing rows, and outliers in key numeric columns."""
    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates()

    # Keep only rows with core required fields.
    core_columns = [
        "customer_key",
        "product_key",
        "sales_amount",
        "order_quantity",
        "list_price",
        "order_date",
    ]
    cleaned = cleaned.dropna(subset=core_columns)

    # Remove outliers in a simple and explainable way using IQR rule.
    numeric_cols = ["sales_amount", "order_quantity", "list_price"]
    for col in numeric_cols:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        cleaned = cleaned[(cleaned[col] >= low) & (cleaned[col] <= high)]

    return cleaned.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, output_path: Path = Path("data/processed/merged_sales_data.csv")) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
