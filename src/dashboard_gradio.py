from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import gradio as gr
import matplotlib
import pandas as pd

from predict_utils import predict_buy, predict_sales

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDA_DIR = Path("outputs/eda")
PROCESSED_DATA_PATH = Path("data/processed/merged_sales_data.csv")

GROUP_COLUMNS = [
    "country_region",
    "region",
    "category",
    "subcategory",
    "product",
    "customer",
    "month",
    "fiscal_quarter",
    "fiscal_year",
    "channel",
]

METRIC_OPTIONS = {
    "Total sales amount": ("sales_amount", "sum"),
    "Average sales amount": ("sales_amount", "mean"),
    "Total order quantity": ("order_quantity", "sum"),
    "Number of order lines": ("sales_order_line_key", "count"),
}


def load_summary() -> str:
    summary_path = EDA_DIR / "summary.json"
    if not summary_path.exists():
        return "Run EDA first: python3 src/data_exploration.py"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [f"- {k}: {v}" for k, v in data.items()]
    return "\n".join(lines)


def load_analysis_data() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        return pd.DataFrame()

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    return df


def get_dropdown_values(column: str) -> list[str]:
    df = load_analysis_data()
    if df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def run_interactive_analysis(
    group_by,
    metric_name,
    country_filter,
    category_filter,
    channel_filter,
    top_n,
):
    df = load_analysis_data()
    if df.empty:
        return pd.DataFrame(), None, "Run EDA first: python3 src/data_exploration.py"

    filtered = df.copy()
    if country_filter:
        filtered = filtered[filtered["country_region"].astype(str).isin(country_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].astype(str).isin(category_filter)]
    if channel_filter:
        filtered = filtered[filtered["channel"].astype(str).isin(channel_filter)]

    if filtered.empty:
        return pd.DataFrame(), None, "No rows match your filters. Try fewer filters."

    if group_by == "customer":
        filtered = filtered[
            filtered["customer"].notna()
            & (filtered["customer"].astype(str).str.strip() != "[Not Applicable]")
        ]
        if filtered.empty:
            return (
                pd.DataFrame(),
                None,
                "No real customer names match your filters. Try fewer filters.",
            )

    metric_col, agg_func = METRIC_OPTIONS[metric_name]
    result = (
        filtered.groupby(group_by, dropna=False)[metric_col]
        .agg(agg_func)
        .reset_index(name=metric_name)
        .sort_values(metric_name, ascending=False)
        .head(int(top_n))
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(result[group_by].astype(str), result[metric_name], color="#2A9D8F")
    ax.set_title(f"{metric_name} by {group_by}")
    ax.set_xlabel(group_by)
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()

    total_sales = filtered["sales_amount"].sum()
    total_quantity = filtered["order_quantity"].sum()
    summary = (
        f"Rows analyzed: {len(filtered):,}\n"
        f"Total sales amount: {total_sales:,.2f}\n"
        f"Total order quantity: {total_quantity:,.0f}"
    )

    return result, fig, summary


def ui_classification_predict(
    customer_key,
    product_key,
    order_quantity,
    list_price,
    country,
    state,
    category,
    subcategory,
    color,
    channel,
    region,
):
    row = {
        "customer_key": int(customer_key),
        "product_key": int(product_key),
        "order_quantity": float(order_quantity),
        "list_price": float(list_price),
        "country_region": country,
        "state_province": state,
        "category": category,
        "subcategory": subcategory,
        "color": color,
        "channel": channel,
        "region": region,
    }

    buy_pred, buy_prob = predict_buy(row)
    buy_label = "Buy" if buy_pred == 1 else "Not Buy"

    return f"{buy_label}", f"{buy_prob:.3f}"


def ui_regression_predict(
    customer_key,
    product_key,
    order_quantity,
    unit_price,
    discount,
    list_price,
    country,
    state,
    category,
    subcategory,
    color,
    channel,
    region,
    month,
    fiscal_quarter,
):
    row = {
        "customer_key": int(customer_key),
        "product_key": int(product_key),
        "order_quantity": float(order_quantity),
        "unit_price": float(unit_price),
        "unit_price_discount_pct": float(discount),
        "list_price": float(list_price),
        "country_region": country,
        "state_province": state,
        "category": category,
        "subcategory": subcategory,
        "color": color,
        "channel": channel,
        "region": region,
        "month": month,
        "fiscal_quarter": fiscal_quarter,
    }

    sales_pred = predict_sales(row)

    return f"{sales_pred:.2f}"


with gr.Blocks(title="CRM Data Analysis Dashboard") as demo:
    gr.Markdown("# CRM Data Analysis Dashboard")
    gr.Markdown(
        "AdventureWorks CRM project: data cleaning, EDA, classification, and regression."
    )

    with gr.Tab("EDA Summary"):
        gr.Markdown("## Data summary")
        summary_box = gr.Textbox(label="Summary", value=load_summary(), lines=8)
        gr.Markdown("## Charts")
        gr.Image(value=str(EDA_DIR / "sales_by_month.png"), label="Monthly Sales")
        gr.Image(value=str(EDA_DIR / "sales_by_region.png"), label="Sales by Region")
        gr.Image(value=str(EDA_DIR / "top_products.png"), label="Top Products")
        gr.Image(value=str(EDA_DIR / "top_customers.png"), label="Top Customers")

    with gr.Tab("Interactive Data Analysis"):
        gr.Markdown("## Explore Current Data")
        gr.Markdown(
            "Choose filters and a grouping to create your own CRM analysis from the cleaned data."
        )
        with gr.Row():
            group_by = gr.Dropdown(
                label="Group data by",
                choices=GROUP_COLUMNS,
                value="category",
            )
            metric_name = gr.Dropdown(
                label="Metric",
                choices=list(METRIC_OPTIONS.keys()),
                value="Total sales amount",
            )
            top_n = gr.Slider(
                label="How many rows to show",
                minimum=5,
                maximum=30,
                step=1,
                value=10,
            )
        with gr.Row():
            country_filter = gr.Dropdown(
                label="Filter by country_region",
                choices=get_dropdown_values("country_region"),
                multiselect=True,
            )
            category_filter = gr.Dropdown(
                label="Filter by category",
                choices=get_dropdown_values("category"),
                multiselect=True,
            )
            channel_filter = gr.Dropdown(
                label="Filter by channel",
                choices=get_dropdown_values("channel"),
                multiselect=True,
            )

        analysis_btn = gr.Button("Run analysis")
        analysis_summary = gr.Textbox(label="Analysis summary", lines=4)
        analysis_plot = gr.Plot(label="Analysis chart")
        analysis_table = gr.Dataframe(label="Analysis table")

        analysis_btn.click(
            fn=run_interactive_analysis,
            inputs=[
                group_by,
                metric_name,
                country_filter,
                category_filter,
                channel_filter,
                top_n,
            ],
            outputs=[analysis_table, analysis_plot, analysis_summary],
        )

    with gr.Tab("Classification: Buy / Not Buy"):
        gr.Markdown("## Pipeline 1: RandomForestClassifier")
        gr.Markdown(
            "Goal: predict whether a customer will buy a selected product."
        )
        with gr.Row():
            cls_customer_key = gr.Number(label="customer_key", value=11000)
            cls_product_key = gr.Number(label="product_key", value=214)
            cls_order_quantity = gr.Number(label="order_quantity", value=1)
        with gr.Row():
            cls_list_price = gr.Number(label="list_price", value=1200)
            cls_country = gr.Textbox(label="country_region", value="United States")
            cls_state = gr.Textbox(label="state_province", value="California")
        with gr.Row():
            cls_category = gr.Textbox(label="category", value="Bikes")
            cls_subcategory = gr.Textbox(label="subcategory", value="Mountain Bikes")
            cls_color = gr.Textbox(label="color", value="Black")
        with gr.Row():
            cls_channel = gr.Textbox(label="channel", value="Reseller")
            cls_region = gr.Textbox(label="region", value="Northwest")

        cls_run_btn = gr.Button("Run classification")
        cls_label_out = gr.Textbox(label="Prediction")
        cls_probability_out = gr.Textbox(label="Probability of Buy")

        cls_run_btn.click(
            fn=ui_classification_predict,
            inputs=[
                cls_customer_key,
                cls_product_key,
                cls_order_quantity,
                cls_list_price,
                cls_country,
                cls_state,
                cls_category,
                cls_subcategory,
                cls_color,
                cls_channel,
                cls_region,
            ],
            outputs=[cls_label_out, cls_probability_out],
        )

    with gr.Tab("Regression: Sales Amount"):
        gr.Markdown("## Pipeline 2: RandomForestRegressor")
        gr.Markdown(
            "Goal: predict the expected sales amount using product, region, and other data."
        )
        with gr.Row():
            reg_customer_key = gr.Number(label="customer_key", value=11000)
            reg_product_key = gr.Number(label="product_key", value=214)
            reg_order_quantity = gr.Number(label="order_quantity", value=1)
        with gr.Row():
            reg_unit_price = gr.Number(label="unit_price", value=1000)
            reg_discount = gr.Number(label="unit_price_discount_pct", value=0.0)
            reg_list_price = gr.Number(label="list_price", value=1200)
        with gr.Row():
            reg_country = gr.Textbox(label="country_region", value="United States")
            reg_state = gr.Textbox(label="state_province", value="California")
            reg_region = gr.Textbox(label="region", value="Northwest")
        with gr.Row():
            reg_category = gr.Textbox(label="category", value="Bikes")
            reg_subcategory = gr.Textbox(label="subcategory", value="Mountain Bikes")
            reg_color = gr.Textbox(label="color", value="Black")
        with gr.Row():
            reg_channel = gr.Textbox(label="channel", value="Reseller")
            reg_month = gr.Textbox(label="month", value="January")
            reg_fiscal_quarter = gr.Textbox(label="fiscal_quarter", value="Q1")

        reg_run_btn = gr.Button("Run regression")
        reg_sales_out = gr.Textbox(label="Predicted sales_amount")

        reg_run_btn.click(
            fn=ui_regression_predict,
            inputs=[
                reg_customer_key,
                reg_product_key,
                reg_order_quantity,
                reg_unit_price,
                reg_discount,
                reg_list_price,
                reg_country,
                reg_state,
                reg_category,
                reg_subcategory,
                reg_color,
                reg_channel,
                reg_region,
                reg_month,
                reg_fiscal_quarter,
            ],
            outputs=[reg_sales_out],
        )


if __name__ == "__main__":
    demo.launch(server_port=7961)
