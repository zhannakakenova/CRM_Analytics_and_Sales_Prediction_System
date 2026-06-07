from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import gradio as gr
import matplotlib
import pandas as pd
import plotly.graph_objects as go

from app.api.services.data_service import grouped_sales, load_eda_summary, load_sales_data, monthly_sales
from app.core import EDA_DIR
from app.dashboard.tableau_embed import tableau_iframe
from app.ml.classifier.service import predict_buy
from app.ml.forecasting.service import forecast_monthly_sales
from app.ml.regressor.service import predict_sales

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _summary_text() -> str:
    summary = load_eda_summary()
    if not summary:
        return "Run EDA first: python3 -m app.api.services.eda_pipeline"
    return "\n".join(f"- {key}: {value}" for key, value in summary.items())


def _line_plot(rows: list[dict], x_column: str, y_column: str, title: str):
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame[x_column], frame[y_column], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def _bar_plot(rows: list[dict], x_column: str, y_column: str, title: str):
    frame = pd.DataFrame(rows).head(12)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(frame[x_column].astype(str), frame[y_column], color="#2A9D8F")
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def region_sales_table():
    rows = grouped_sales("region")
    return pd.DataFrame(rows), _bar_plot(rows, "region", "sales_amount", "Sales by Region")


def category_sales_table():
    rows = grouped_sales("category")
    return pd.DataFrame(rows), _bar_plot(rows, "category", "sales_amount", "Sales by Category")


def run_interactive_analysis(
    group_by,
    metric_name,
    country_filter,
    category_filter,
    channel_filter,
    top_n,
):
    df = load_sales_data()
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
            return pd.DataFrame(), None, "No real customer names match your filters."

    metric_column, aggregate_function = METRIC_OPTIONS[metric_name]
    result = (
        filtered.groupby(group_by, dropna=False)[metric_column]
        .agg(aggregate_function)
        .reset_index(name=metric_name)
        .sort_values(metric_name, ascending=False)
        .head(int(top_n))
    )

    fig = _bar_plot(result.to_dict(orient="records"), group_by, metric_name, f"{metric_name} by {group_by}")
    summary = (
        f"Rows analyzed: {len(filtered):,}\n"
        f"Total sales amount: {filtered['sales_amount'].sum():,.2f}\n"
        f"Total order quantity: {filtered['order_quantity'].sum():,.0f}"
    )
    return result, fig, summary


def buy_prediction(
    customer_key,
    product_key,
    order_quantity,
    list_price,
    country_region,
    state_province,
    category,
    subcategory,
    color,
    channel,
    region,
):
    payload = {
        "customer_key": int(customer_key),
        "product_key": int(product_key),
        "order_quantity": float(order_quantity),
        "list_price": float(list_price),
        "country_region": country_region,
        "state_province": state_province,
        "category": category,
        "subcategory": subcategory,
        "color": color,
        "channel": channel,
        "region": region,
    }
    prediction, probability = predict_buy(payload)
    return prediction, f"{probability:.3f}"


def sales_prediction(
    customer_key,
    product_key,
    order_quantity,
    unit_price,
    discount,
    list_price,
    country_region,
    state_province,
    category,
    subcategory,
    color,
    channel,
    region,
    month,
    fiscal_quarter,
):
    payload = {
        "customer_key": int(customer_key),
        "product_key": int(product_key),
        "order_quantity": float(order_quantity),
        "unit_price": float(unit_price),
        "unit_price_discount_pct": float(discount),
        "list_price": float(list_price),
        "country_region": country_region,
        "state_province": state_province,
        "category": category,
        "subcategory": subcategory,
        "color": color,
        "channel": channel,
        "region": region,
        "month": month,
        "fiscal_quarter": fiscal_quarter,
    }
    return f"{predict_sales(payload):.2f}"


def forecast_ui(product, region, horizon):
    result = forecast_monthly_sales(
        product=product or None,
        region=region or None,
        horizon=int(horizon),
    )
    frame = pd.DataFrame(result["forecast"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["predicted_sales_amount"],
            mode="lines+markers",
            name="Predicted sales",
        )
    )
    fig.update_layout(
        title="Future Sales Forecast",
        xaxis_title="Month",
        yaxis_title="Predicted Sales Amount",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    summary = (
        f"Predicted Sales Next Month: {result['predicted_sales_next_month']:,.2f}\n"
        f"Predicted Sales Next Quarter: {result['predicted_sales_next_quarter']:,.2f}"
    )
    return summary, frame, fig


def dropdown_values(column: str) -> list[str]:
    frame = load_sales_data()
    return sorted(frame[column].dropna().astype(str).unique().tolist())


with gr.Blocks(title="AdventureWorks Analytics Platform") as demo:
    gr.Markdown("# AdventureWorks Analytics Platform")

    with gr.Tab("Tableau Dashboard"):
        gr.HTML(tableau_iframe())

    with gr.Tab("EDA Summary"):
        gr.Textbox(label="Summary", value=_summary_text(), lines=8)
        monthly_rows = monthly_sales()
        gr.Plot(value=_line_plot(monthly_rows, "month", "sales_amount", "Monthly Sales"))
        region_table, region_plot = region_sales_table()
        gr.Plot(value=region_plot)
        gr.Dataframe(value=region_table)
        with gr.Row():
            gr.Image(value=str(EDA_DIR / "top_products.png"), label="Top Products")
            gr.Image(value=str(EDA_DIR / "top_customers.png"), label="Top Customers")

    with gr.Tab("Interactive Analysis"):
        with gr.Row():
            group_by = gr.Dropdown(label="Group data by", choices=GROUP_COLUMNS, value="category")
            metric_name = gr.Dropdown(
                label="Metric",
                choices=list(METRIC_OPTIONS.keys()),
                value="Total sales amount",
            )
            top_n = gr.Slider(label="Rows to show", minimum=5, maximum=30, step=1, value=10)
        with gr.Row():
            country_filter = gr.Dropdown(
                label="Filter by country_region",
                choices=dropdown_values("country_region"),
                multiselect=True,
            )
            category_filter = gr.Dropdown(
                label="Filter by category",
                choices=dropdown_values("category"),
                multiselect=True,
            )
            channel_filter = gr.Dropdown(
                label="Filter by channel",
                choices=dropdown_values("channel"),
                multiselect=True,
            )
        analysis_button = gr.Button("Run analysis")
        analysis_summary = gr.Textbox(label="Analysis summary", lines=4)
        analysis_plot = gr.Plot(label="Analysis chart")
        analysis_table = gr.Dataframe(label="Analysis table")
        analysis_button.click(
            run_interactive_analysis,
            [group_by, metric_name, country_filter, category_filter, channel_filter, top_n],
            [analysis_table, analysis_plot, analysis_summary],
        )

    with gr.Tab("Buy / Not Buy Prediction"):
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
        cls_button = gr.Button("Run classification")
        cls_prediction = gr.Textbox(label="Prediction")
        cls_probability = gr.Textbox(label="Probability of Buy")
        cls_button.click(
            buy_prediction,
            [
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
            [cls_prediction, cls_probability],
        )

    with gr.Tab("Sales Prediction"):
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
        reg_button = gr.Button("Run regression")
        reg_output = gr.Textbox(label="Predicted sales_amount")
        reg_button.click(
            sales_prediction,
            [
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
            reg_output,
        )

    with gr.Tab("Future Sales Forecast"):
        with gr.Row():
            forecast_product = gr.Dropdown(label="Product", choices=dropdown_values("product"))
            forecast_region = gr.Dropdown(label="Region", choices=dropdown_values("region"))
            forecast_horizon = gr.Slider(label="Forecast Horizon", minimum=1, maximum=24, step=1, value=6)
        forecast_button = gr.Button("Run forecast")
        forecast_summary = gr.Textbox(label="Forecast summary", lines=2)
        forecast_table = gr.Dataframe(label="Forecast table")
        forecast_plot = gr.Plot(label="Forecast chart")
        forecast_button.click(
            forecast_ui,
            [forecast_product, forecast_region, forecast_horizon],
            [forecast_summary, forecast_table, forecast_plot],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
