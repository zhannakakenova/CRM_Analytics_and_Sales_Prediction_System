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

BRAND_NAVY = "#10243E"
BRAND_BLUE = "#2563EB"
BRAND_TEAL = "#0F9F8F"
TEXT_MUTED = "#64748B"
APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
)

APP_CSS = """
:root {
    --brand-navy: #10243e;
    --brand-blue: #2563eb;
    --brand-teal: #0f9f8f;
    --surface: #ffffff;
    --canvas: #f3f6fa;
    --border: #e3e9f1;
    --muted: #64748b;
    color-scheme: light;
}

html {
    width: 100%;
    min-width: 100%;
    overflow-x: hidden;
    overflow-y: scroll;
    scrollbar-gutter: stable;
    background: var(--canvas) !important;
}

body {
    width: 100%;
    min-width: 100%;
    margin: 0 !important;
    overflow-x: hidden;
}

html, body, .gradio-container {
    background: var(--canvas) !important;
    color: var(--brand-navy);
    color-scheme: light !important;
}

.gradio-container * {
    box-sizing: border-box;
}

.gradio-container {
    width: min(100%, 1480px) !important;
    min-width: 0 !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 22px 28px 48px !important;
    overflow-x: hidden;
}

.gradio-container,
.gradio-container > *,
.gradio-container [role="tabpanel"] {
    transition: none !important;
    animation: none !important;
}

.app-hero {
    position: relative;
    overflow: hidden;
    padding: 32px 36px;
    margin-bottom: 18px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    background:
        radial-gradient(circle at 88% 12%, rgba(45, 212, 191, 0.24), transparent 28%),
        linear-gradient(135deg, #0b1c33 0%, #173d67 58%, #145c69 100%);
    color: white;
    box-shadow: 0 24px 55px rgba(15, 35, 62, 0.18);
}

.app-hero__eyebrow {
    margin-bottom: 12px;
    color: #99f6e4;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.app-hero h1 {
    margin: 0 0 10px;
    color: white;
    font-size: clamp(28px, 4vw, 46px);
    line-height: 1.05;
    letter-spacing: -.04em;
}

.app-hero p {
    max-width: 720px;
    margin: 0;
    color: #dbeafe;
    font-size: 16px;
    line-height: 1.65;
}

.app-hero__status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 22px;
    padding: 8px 12px;
    border: 1px solid rgba(255, 255, 255, .16);
    border-radius: 999px;
    background: rgba(255, 255, 255, .08);
    color: #e0f2fe;
    font-size: 12px;
    font-weight: 700;
}

.app-hero__status::before {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #5eead4;
    box-shadow: 0 0 0 5px rgba(94, 234, 212, .12);
    content: "";
}

.section-heading {
    padding: 4px 2px 10px;
}

.section-heading h2 {
    margin: 0 0 5px;
    color: var(--brand-navy);
    font-size: 22px;
    letter-spacing: -.025em;
}

.section-heading p {
    margin: 0;
    color: var(--muted);
    font-size: 14px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 6px 0 18px;
}

.metric-card {
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--surface);
    box-shadow: 0 10px 24px rgba(15, 35, 62, .06);
}

.metric-card__label {
    margin-bottom: 8px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.metric-card__value {
    color: var(--brand-navy);
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -.03em;
    line-height: 1.15;
    overflow-wrap: anywhere;
}

.panel, .form-panel, .result-panel {
    padding: 18px !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    background: var(--surface) !important;
    box-shadow: 0 10px 28px rgba(15, 35, 62, .06) !important;
}

.form-panel { background: #f8fafc !important; }
.result-panel { border-color: #cbdff8 !important; background: #f7fbff !important; }

.primary-action {
    min-height: 46px !important;
    border: 0 !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, var(--brand-blue), #1877d3) !important;
    box-shadow: 0 10px 18px rgba(37, 99, 235, .18) !important;
    font-weight: 800 !important;
}

.gradio-container [role="tablist"] {
    display: flex !important;
    gap: 5px !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding: 5px !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: white !important;
    box-shadow: 0 8px 20px rgba(15, 35, 62, .05);
    scrollbar-width: thin;
}

.gradio-container [role="tab"] {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    padding: 9px 13px !important;
    border-radius: 9px !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
}

.gradio-container [role="tab"][aria-selected="true"] {
    background: var(--brand-navy) !important;
    color: white !important;
}

.gradio-container [role="tabpanel"],
.stable-layout,
.stable-layout > *,
.form-panel .row,
.form-panel .row > * {
    min-width: 0 !important;
}

.gradio-container [role="tabpanel"] {
    width: 100% !important;
    min-height: 720px;
    overflow: visible !important;
}

.stable-layout {
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
    gap: 18px !important;
}

.stable-layout > .form-panel {
    flex: 1 1 auto !important;
    width: calc(100% - 338px) !important;
    max-width: calc(100% - 338px) !important;
}

.action-panel {
    position: sticky;
    top: 16px;
    flex: 0 0 320px !important;
    width: 320px !important;
    max-width: 320px !important;
    padding: 18px !important;
    border: 1px solid #cbdff8 !important;
    border-radius: 18px !important;
    background: #f7fbff !important;
    box-shadow: 0 10px 28px rgba(15, 35, 62, .06) !important;
}

.action-panel .result-panel {
    padding: 12px !important;
    box-shadow: none !important;
}

.action-panel,
.action-panel h3,
.action-panel p,
.action-panel label,
.result-panel,
.result-panel label {
    color: var(--brand-navy) !important;
}

.form-panel input,
.form-panel textarea,
.form-panel select,
.result-panel input,
.result-panel textarea {
    color-scheme: light !important;
}

.tableau-shell iframe {
    display: block;
    max-width: 100%;
    border-radius: 16px;
    background: white;
}

footer { display: none !important; }

@media (max-width: 1050px) {
    .gradio-container {
        width: 100% !important;
        padding: 12px 12px 32px !important;
    }
    .app-hero { padding: 26px 22px; border-radius: 18px; }
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .stable-layout { flex-wrap: wrap !important; }
    .stable-layout > .form-panel,
    .action-panel {
        position: static;
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
    }
}

@media (max-width: 620px) {
    .metric-grid { grid-template-columns: 1fr; }
    .action-panel { position: static; }
}
"""

HERO_HTML = """
<section class="app-hero">
    <div class="app-hero__eyebrow">AdventureWorks · Revenue Intelligence</div>
    <h1>Sales command center</h1>
    <p>Explore performance, uncover commercial patterns, and turn historical sales data into confident next actions.</p>
    <div class="app-hero__status">Analytics workspace online</div>
</section>
"""

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


def _summary_cards() -> str:
    summary = load_eda_summary()
    if not summary:
        return '<div class="metric-card">Run EDA to generate the executive overview.</div>'
    date_range = summary.get("date_range", ["—", "—"])
    metrics = [
        ("Revenue", f"${summary.get('total_sales', 0):,.0f}"),
        ("Transactions", f"{summary.get('rows_after_cleaning', 0):,}"),
        ("Average sale", f"${summary.get('average_sales', 0):,.2f}"),
        ("Reporting period", f"{date_range[0]} — {date_range[-1]}"),
    ]
    cards = "".join(
        f'<article class="metric-card"><div class="metric-card__label">{label}</div>'
        f'<div class="metric-card__value">{value}</div></article>'
        for label, value in metrics
    )
    return f'<section class="metric-grid">{cards}</section>'


def _style_matplotlib_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color="#E8EEF5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.title.set_color(BRAND_NAVY)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)


def _line_plot(rows: list[dict], x_column: str, y_column: str, title: str):
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#FFFFFF")
    ax.plot(frame[x_column], frame[y_column], marker="o", color=BRAND_BLUE, linewidth=2.4)
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=35)
    _style_matplotlib_axes(ax)
    fig.tight_layout()
    return fig


def _bar_plot(rows: list[dict], x_column: str, y_column: str, title: str):
    frame = pd.DataFrame(rows).head(12)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#FFFFFF")
    ax.bar(frame[x_column].astype(str), frame[y_column], color=BRAND_TEAL, width=0.68)
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=35)
    _style_matplotlib_axes(ax)
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
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": TEXT_MUTED},
        title_font={"color": BRAND_NAVY, "size": 18},
        colorway=[BRAND_BLUE],
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E8EEF5", zeroline=False)
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


with gr.Blocks(title="AdventureWorks Analytics Platform", fill_width=True) as demo:
    gr.HTML(HERO_HTML)

    with gr.Tab("Dashboard"):
        gr.HTML(
            '<div class="section-heading"><h2>Executive dashboard</h2>'
            "<p>A complete Tableau view of revenue performance and commercial momentum.</p></div>"
        )
        gr.HTML(tableau_iframe(), elem_classes="tableau-shell")

    with gr.Tab("Overview"):
        gr.HTML(
            '<div class="section-heading"><h2>Performance overview</h2>'
            "<p>Core KPIs and historical patterns from the prepared sales dataset.</p></div>"
        )
        gr.HTML(_summary_cards())
        monthly_rows = monthly_sales()
        with gr.Row():
            gr.Plot(
                value=_line_plot(monthly_rows, "month", "sales_amount", "Monthly Sales"),
                elem_classes="panel",
            )
            region_table, region_plot = region_sales_table()
            gr.Plot(value=region_plot, elem_classes="panel")
        gr.Dataframe(value=region_table, elem_classes="panel")
        with gr.Row():
            gr.Image(value=str(EDA_DIR / "top_products.png"), label="Top Products", elem_classes="panel")
            gr.Image(value=str(EDA_DIR / "top_customers.png"), label="Top Customers", elem_classes="panel")

    with gr.Tab("Analysis"):
        gr.HTML(
            '<div class="section-heading"><h2>Analysis studio</h2>'
            "<p>Build a focused performance cut using dimensions, metrics, and commercial filters.</p></div>"
        )
        with gr.Group(elem_classes="form-panel"):
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
                    label="Country",
                    choices=dropdown_values("country_region"),
                    multiselect=True,
                )
                category_filter = gr.Dropdown(
                    label="Category",
                    choices=dropdown_values("category"),
                    multiselect=True,
                )
                channel_filter = gr.Dropdown(
                    label="Channel",
                    choices=dropdown_values("channel"),
                    multiselect=True,
                )
            analysis_button = gr.Button("Generate analysis", variant="primary", elem_classes="primary-action")
        analysis_summary = gr.Textbox(label="Analysis summary", lines=4, elem_classes="result-panel")
        analysis_plot = gr.Plot(label="Analysis chart", elem_classes="panel")
        analysis_table = gr.Dataframe(label="Analysis table", elem_classes="panel")
        analysis_button.click(
            run_interactive_analysis,
            [group_by, metric_name, country_filter, category_filter, channel_filter, top_n],
            [analysis_table, analysis_plot, analysis_summary],
        )

    with gr.Tab("Propensity"):
        gr.HTML(
            '<div class="section-heading"><h2>Purchase propensity</h2>'
            "<p>Estimate the likelihood of a customer purchasing a selected product.</p></div>"
        )
        with gr.Row(elem_classes="stable-layout"):
            with gr.Column(scale=3, min_width=520, elem_classes="form-panel"):
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
                with gr.Row():
                    cls_country = gr.Dropdown(label="Country", choices=COUNTRY_VALUES, value=DEFAULT_CUSTOMER[0])
                    cls_state = gr.Dropdown(
                        label="State / province",
                        choices=dropdown_values_for("country_region", DEFAULT_CUSTOMER[0], "state_province"),
                        value=DEFAULT_CUSTOMER[1].value,
                    )
                    cls_channel = gr.Dropdown(label="Channel", choices=CHANNEL_VALUES, value=DEFAULT_CUSTOMER[3])
                with gr.Row():
                    cls_category = gr.Dropdown(label="Category", choices=CATEGORY_VALUES, value=DEFAULT_PRODUCT[0])
                    cls_subcategory = gr.Dropdown(
                        label="Subcategory",
                        choices=dropdown_values_for("category", DEFAULT_PRODUCT[0], "subcategory"),
                        value=DEFAULT_PRODUCT[1].value,
                    )
                with gr.Row():
                    cls_color = gr.Dropdown(label="Color", choices=COLOR_VALUES, value=DEFAULT_PRODUCT[2])
                    cls_list_price = gr.Number(label="List price", value=DEFAULT_PRODUCT[3], minimum=0)
            with gr.Column(scale=1, min_width=280, elem_classes="action-panel"):
                gr.Markdown("### Prediction result\nSelect a customer and product, then run the model.")
                cls_button = gr.Button(
                    "Calculate purchase propensity",
                    variant="primary",
                    elem_classes="primary-action",
                )
                cls_prediction = gr.Textbox(label="Decision", elem_classes="result-panel")
                cls_probability = gr.Textbox(label="Buy probability", elem_classes="result-panel")
        cls_region = gr.State(DEFAULT_CUSTOMER[2])
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

    with gr.Tab("Deal value"):
        gr.HTML(
            '<div class="section-heading"><h2>Deal value prediction</h2>'
            "<p>Estimate expected transaction value using customer, product, and order context.</p></div>"
        )
        with gr.Row(elem_classes="stable-layout"):
            with gr.Column(scale=3, min_width=520, elem_classes="form-panel"):
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
                with gr.Row():
                    reg_order_quantity = gr.Slider(label="Order quantity", minimum=1, maximum=3, step=1, value=1)
                    reg_discount = gr.Slider(
                        label="Discount percentage",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.0,
                    )
                with gr.Row():
                    reg_unit_price = gr.Number(label="Unit price", value=DEFAULT_PRODUCT[3], minimum=0)
                    reg_list_price = gr.Number(label="List price", value=DEFAULT_PRODUCT[3], minimum=0)
                with gr.Row():
                    reg_country = gr.Dropdown(label="Country", choices=COUNTRY_VALUES, value=DEFAULT_CUSTOMER[0])
                    reg_state = gr.Dropdown(
                        label="State / province",
                        choices=dropdown_values_for("country_region", DEFAULT_CUSTOMER[0], "state_province"),
                        value=DEFAULT_CUSTOMER[1].value,
                    )
                    reg_channel = gr.Dropdown(label="Channel", choices=CHANNEL_VALUES, value=DEFAULT_CUSTOMER[3])
                with gr.Row():
                    reg_category = gr.Dropdown(label="Category", choices=CATEGORY_VALUES, value=DEFAULT_PRODUCT[0])
                    reg_subcategory = gr.Dropdown(
                        label="Subcategory",
                        choices=dropdown_values_for("category", DEFAULT_PRODUCT[0], "subcategory"),
                        value=DEFAULT_PRODUCT[1].value,
                    )
                    reg_color = gr.Dropdown(label="Color", choices=COLOR_VALUES, value=DEFAULT_PRODUCT[2])
                with gr.Row():
                    reg_month = gr.Dropdown(label="Month", choices=MONTH_VALUES, value=MONTH_VALUES[0])
                    reg_fiscal_quarter = gr.Dropdown(
                        label="Fiscal quarter",
                        choices=FISCAL_QUARTER_VALUES,
                        value=FISCAL_QUARTER_VALUES[0],
                    )
            with gr.Column(scale=1, min_width=280, elem_classes="action-panel"):
                gr.Markdown("### Predicted value\nReview the populated deal context, then run the model.")
                reg_button = gr.Button("Predict deal value", variant="primary", elem_classes="primary-action")
                reg_output = gr.Textbox(label="Predicted sales amount", elem_classes="result-panel")
        reg_region = gr.State(DEFAULT_CUSTOMER[2])
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

    with gr.Tab("Forecast"):
        gr.HTML(
            '<div class="section-heading"><h2>Revenue forecast</h2>'
            "<p>Project future sales by product, region, and planning horizon.</p></div>"
        )
        with gr.Row(elem_classes="form-panel"):
            forecast_product = gr.Dropdown(label="Product", choices=dropdown_values("product"))
            forecast_region = gr.Dropdown(label="Region", choices=dropdown_values("region"))
            forecast_horizon = gr.Slider(label="Forecast Horizon", minimum=1, maximum=24, step=1, value=6)
        forecast_button = gr.Button("Generate revenue forecast", variant="primary", elem_classes="primary-action")
        forecast_summary = gr.Textbox(label="Forecast summary", lines=2, elem_classes="result-panel")
        forecast_table = gr.Dataframe(label="Forecast table", elem_classes="panel")
        forecast_plot = gr.Plot(label="Forecast chart", elem_classes="panel")
        forecast_button.click(
            forecast_ui,
            [forecast_product, forecast_region, forecast_horizon],
            [forecast_summary, forecast_table, forecast_plot],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=APP_CSS,
        theme=APP_THEME,
    )
