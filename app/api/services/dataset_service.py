from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core import DATA_PATH, RAW_DATA_DIR


def load_tables(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    return {
        "sales": pd.read_csv(data_dir / "sales_data.csv"),
        "orders": pd.read_csv(data_dir / "sales_order_data.csv"),
        "customers": pd.read_csv(data_dir / "customer_data.csv"),
        "products": pd.read_csv(data_dir / "product_data.csv"),
        "territories": pd.read_csv(data_dir / "sales_territory_data.csv"),
        "dates": pd.read_csv(data_dir / "date_data.csv"),
    }


def build_merged_dataset(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    tables = load_tables(data_dir)

    df = tables["sales"].merge(tables["orders"], on="sales_order_line_key", how="left")
    df = df.merge(tables["customers"], on="customer_key", how="left")
    df = df.merge(tables["products"], on="product_key", how="left")
    df = df.merge(tables["territories"], on="sales_territory_key", how="left")

    date_info = tables["dates"][["date_key", "full_date", "month", "fiscal_quarter", "fiscal_year"]].copy()
    df = df.merge(date_info, left_on="order_date_key", right_on="date_key", how="left")
    df = df.rename(columns={"full_date": "order_date"})
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy().drop_duplicates()
    core_columns = [
        "customer_key",
        "product_key",
        "sales_amount",
        "order_quantity",
        "list_price",
        "order_date",
    ]
    cleaned = cleaned.dropna(subset=core_columns)

    for column in ["sales_amount", "order_quantity", "list_price"]:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        cleaned = cleaned[(cleaned[column] >= low) & (cleaned[column] <= high)]

    return cleaned.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, output_path: Path = DATA_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
