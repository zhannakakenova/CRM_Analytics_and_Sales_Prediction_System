from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

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
    try:
        prediction, probability = predict_buy(payload)
    except Exception as error:
        return f"Error: {error}", "Check terminal logs"
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
    try:
        return f"{predict_sales(payload):.2f}"
    except Exception as error:
        return f"Error: {error}"


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


PLACEHOLDER_VALUES = {"", "[Not Applicable]", "Unknown", "nan"}


def _clean_dropdown_values(values: pd.Series) -> list[str]:
    unique_values = values.dropna().astype(str).str.strip().unique().tolist()
    return sorted(value for value in unique_values if value not in PLACEHOLDER_VALUES)


def dropdown_values(column: str) -> list[str]:
    frame = load_sales_data()
    return _clean_dropdown_values(frame[column])


def dropdown_values_for(parent_column: str, parent_value, child_column: str) -> list[str]:
    frame = load_sales_data()
    filtered = frame[frame[parent_column].astype(str) == str(parent_value)]
    return _clean_dropdown_values(filtered[child_column])


def customer_choices() -> list[tuple[str, int]]:
    frame = load_sales_data()
    customers = (
        frame[["customer_key", "customer"]]
        .dropna()
        .drop_duplicates()
        .query("customer_key >= 0")
        .sort_values(["customer", "customer_key"])
    )
    return [
        (f"{row.customer} (#{int(row.customer_key)})", int(row.customer_key))
        for row in customers.itertuples(index=False)
    ]


def product_choices() -> list[tuple[str, int]]:
    frame = load_sales_data()
    products = (
        frame[["product_key", "product"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["product", "product_key"])
    )
    return [
        (f"{row.product} (#{int(row.product_key)})", int(row.product_key))
        for row in products.itertuples(index=False)
    ]


def sales_territory(country_region: str, state_province: str) -> str:
    frame = load_sales_data()
    matches = frame[
        (frame["country_region"].astype(str) == str(country_region))
        & (frame["state_province"].astype(str) == str(state_province))
    ]
    if matches.empty:
        return ""
    return str(matches["region"].mode().iloc[0])


def update_states_and_territory(country_region: str):
    choices = dropdown_values_for("country_region", country_region, "state_province")
    state_province = choices[0] if choices else None
    territory = sales_territory(country_region, state_province) if state_province else ""
    return gr.Dropdown(choices=choices, value=state_province), territory


def update_territory(country_region: str, state_province: str) -> str:
    return sales_territory(country_region, state_province)


def update_subcategories(category: str):
    choices = dropdown_values_for("category", category, "subcategory")
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def customer_details(customer_key: int):
    frame = load_sales_data()
    row = frame[frame["customer_key"] == int(customer_key)].iloc[0]
    state_choices = dropdown_values_for("country_region", row["country_region"], "state_province")
    return (
        row["country_region"],
        gr.Dropdown(choices=state_choices, value=row["state_province"]),
        row["region"],
        row["channel"],
    )


def product_details(product_key: int):
    frame = load_sales_data()
    row = frame[frame["product_key"] == int(product_key)].iloc[0]
    subcategory_choices = dropdown_values_for("category", row["category"], "subcategory")
    return (
        row["category"],
        gr.Dropdown(choices=subcategory_choices, value=row["subcategory"]),
        row["color"],
        float(row["list_price"]),
    )


def regression_product_details(product_key: int):
    category, subcategory, color, list_price = product_details(product_key)
    return category, subcategory, color, list_price, list_price


DEFAULT_CUSTOMER_KEY = 11000
DEFAULT_PRODUCT_KEY = 214
DEFAULT_CUSTOMER = customer_details(DEFAULT_CUSTOMER_KEY)
DEFAULT_PRODUCT = product_details(DEFAULT_PRODUCT_KEY)
COUNTRY_VALUES = dropdown_values("country_region")
CATEGORY_VALUES = dropdown_values("category")
COLOR_VALUES = dropdown_values("color")
CHANNEL_VALUES = dropdown_values("channel")
MONTH_VALUES = dropdown_values("month")
FISCAL_QUARTER_VALUES = dropdown_values("fiscal_quarter")


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
        gr.Markdown("Select a customer and product to automatically fill their known attributes.")
        with gr.Row():
            cls_customer_key = gr.Dropdown(
                label="Customer",
                choices=customer_choices(),
                value=DEFAULT_CUSTOMER_KEY,
                filterable=True,
            )
            cls_product_key = gr.Dropdown(
                label="Product",
                choices=product_choices(),
                value=DEFAULT_PRODUCT_KEY,
                filterable=True,
            )
            cls_order_quantity = gr.Slider(label="Order quantity", minimum=1, maximum=3, step=1, value=1)
        with gr.Row():
            cls_list_price = gr.Number(label="List price", value=DEFAULT_PRODUCT[3], minimum=0)
            cls_country = gr.Dropdown(label="Country", choices=COUNTRY_VALUES, value=DEFAULT_CUSTOMER[0])
            cls_state = gr.Dropdown(
                label="State / province",
                choices=dropdown_values_for("country_region", DEFAULT_CUSTOMER[0], "state_province"),
                value=DEFAULT_CUSTOMER[1].value,
            )
        with gr.Row():
            cls_category = gr.Dropdown(label="Category", choices=CATEGORY_VALUES, value=DEFAULT_PRODUCT[0])
            cls_subcategory = gr.Dropdown(
                label="Subcategory",
                choices=dropdown_values_for("category", DEFAULT_PRODUCT[0], "subcategory"),
                value=DEFAULT_PRODUCT[1].value,
            )
            cls_color = gr.Dropdown(label="Color", choices=COLOR_VALUES, value=DEFAULT_PRODUCT[2])
        with gr.Row():
            cls_channel = gr.Dropdown(label="Channel", choices=CHANNEL_VALUES, value=DEFAULT_CUSTOMER[3])
        cls_region = gr.State(DEFAULT_CUSTOMER[2])
        cls_button = gr.Button("Run classification")
        cls_prediction = gr.Textbox(label="Prediction")
        cls_probability = gr.Textbox(label="Probability of Buy")
        cls_country.change(update_states_and_territory, cls_country, [cls_state, cls_region])
        cls_state.change(update_territory, [cls_country, cls_state], cls_region)
        cls_category.change(update_subcategories, cls_category, cls_subcategory)
        cls_customer_key.change(
            customer_details,
            cls_customer_key,
            [cls_country, cls_state, cls_region, cls_channel],
        )
        cls_product_key.change(
            product_details,
            cls_product_key,
            [cls_category, cls_subcategory, cls_color, cls_list_price],
        )
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
        gr.Markdown("Customer and product selections automatically fill the related categorical fields.")
        with gr.Row():
            reg_customer_key = gr.Dropdown(
                label="Customer",
                choices=customer_choices(),
                value=DEFAULT_CUSTOMER_KEY,
                filterable=True,
            )
            reg_product_key = gr.Dropdown(
                label="Product",
                choices=product_choices(),
                value=DEFAULT_PRODUCT_KEY,
                filterable=True,
            )
            reg_order_quantity = gr.Slider(label="Order quantity", minimum=1, maximum=3, step=1, value=1)
        with gr.Row():
            reg_unit_price = gr.Number(label="Unit price", value=DEFAULT_PRODUCT[3], minimum=0)
            reg_discount = gr.Slider(label="Discount percentage", minimum=0, maximum=1, step=0.01, value=0.0)
            reg_list_price = gr.Number(label="List price", value=DEFAULT_PRODUCT[3], minimum=0)
        with gr.Row():
            reg_country = gr.Dropdown(label="Country", choices=COUNTRY_VALUES, value=DEFAULT_CUSTOMER[0])
            reg_state = gr.Dropdown(
                label="State / province",
                choices=dropdown_values_for("country_region", DEFAULT_CUSTOMER[0], "state_province"),
                value=DEFAULT_CUSTOMER[1].value,
            )
        reg_region = gr.State(DEFAULT_CUSTOMER[2])
        with gr.Row():
            reg_category = gr.Dropdown(label="Category", choices=CATEGORY_VALUES, value=DEFAULT_PRODUCT[0])
            reg_subcategory = gr.Dropdown(
                label="Subcategory",
                choices=dropdown_values_for("category", DEFAULT_PRODUCT[0], "subcategory"),
                value=DEFAULT_PRODUCT[1].value,
            )
            reg_color = gr.Dropdown(label="Color", choices=COLOR_VALUES, value=DEFAULT_PRODUCT[2])
        with gr.Row():
            reg_channel = gr.Dropdown(label="Channel", choices=CHANNEL_VALUES, value=DEFAULT_CUSTOMER[3])
            reg_month = gr.Dropdown(label="Month", choices=MONTH_VALUES, value=MONTH_VALUES[0])
            reg_fiscal_quarter = gr.Dropdown(
                label="Fiscal quarter",
                choices=FISCAL_QUARTER_VALUES,
                value=FISCAL_QUARTER_VALUES[0],
            )
        reg_button = gr.Button("Run regression")
        reg_output = gr.Textbox(label="Predicted sales_amount")
        reg_country.change(update_states_and_territory, reg_country, [reg_state, reg_region])
        reg_state.change(update_territory, [reg_country, reg_state], reg_region)
        reg_category.change(update_subcategories, reg_category, reg_subcategory)
        reg_customer_key.change(
            customer_details,
            reg_customer_key,
            [reg_country, reg_state, reg_region, reg_channel],
        )
        reg_product_key.change(
            regression_product_details,
            reg_product_key,
            [reg_category, reg_subcategory, reg_color, reg_list_price, reg_unit_price],
        )
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
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
