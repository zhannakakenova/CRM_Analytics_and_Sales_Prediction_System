from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import build_merged_dataset, clean_dataset, save_processed_data

EDA_DIR = Path("outputs/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(EDA_DIR / fig_name, dpi=140)
    plt.close()


def run_eda() -> None:
    raw = build_merged_dataset()
    df = clean_dataset(raw)
    save_processed_data(df)

    summary = {
        "rows_after_cleaning": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "date_range": [str(df["order_date"].min().date()), str(df["order_date"].max().date())],
        "total_sales": float(df["sales_amount"].sum()),
        "average_sales": float(df["sales_amount"].mean()),
        "top_country_by_sales": df.groupby("country_region")["sales_amount"].sum().sort_values(ascending=False).head(1).index.tolist(),
    }

    monthly = (
        df.set_index("order_date")
        .resample("M")["sales_amount"]
        .sum()
        .reset_index()
    )
    plt.figure(figsize=(10, 4))
    plt.plot(monthly["order_date"], monthly["sales_amount"])
    plt.title("Monthly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales Amount")
    save_plot("sales_by_month.png")

    top_regions = df.groupby("region")["sales_amount"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    top_regions.plot(kind="bar", color="#2A9D8F")
    plt.title("Top Regions by Sales")
    plt.xlabel("Region")
    plt.ylabel("Sales Amount")
    save_plot("sales_by_region.png")

    top_products = df.groupby("product")["sales_amount"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 4))
    top_products.plot(kind="bar", color="#E76F51")
    plt.title("Top 10 Products by Sales")
    plt.xlabel("Product")
    plt.ylabel("Sales Amount")
    save_plot("top_products.png")

    real_customers = df[
        df["customer"].notna()
        & (df["customer"].astype(str).str.strip() != "[Not Applicable]")
    ]
    top_customers = (
        real_customers.groupby("customer")["sales_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 4))
    top_customers.plot(kind="bar", color="#264653")
    plt.title("Top 10 Customers by Sales")
    plt.xlabel("Customer")
    plt.ylabel("Sales Amount")
    save_plot("top_customers.png")

    with open(EDA_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("EDA finished.")
    print(f"Saved charts to: {EDA_DIR}")
    print("Processed file: data/processed/merged_sales_data.csv")


if __name__ == "__main__":
    run_eda()
