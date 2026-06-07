Task:
FastAPI API (/predict/sales, /eda/...)
Swagger /docs
MVC structure
Pydantic
pytest
async/await
Dockerfile
Add Time Series Forecasting
Given:
tableu  Embed Code:
<div class='tableauPlaceholder' id='viz1780825045832' style='position: relative'><noscript><a href='#'><img alt='Dashboard ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard_Sales_17808249802880&#47;Dashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Dashboard_Sales_17808249802880&#47;Dashboard' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard_Sales_17808249802880&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1780825045832');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1477px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>



# UPDATED TASK: AdventureWorks Analytics Platform

## Architecture

Single Docker container running:

```text
FastAPI
+
Gradio UI
+
Tableau Dashboard
+
RandomForest Classifier
+
RandomForest Regressor
+
Time Series Forecasting
```

---

# User Flow

```text
Gradio Homepage

├── Tableau Dashboard
├── EDA Summary
├── Interactive Analysis
├── Buy / Not Buy Prediction
├── Sales Prediction
└── Future Sales Forecast
```

FastAPI serves:

```text
/docs
/redoc
/api/*
```

Gradio serves:

```text
/
```

---

# Required Features

## 1. FastAPI API

### EDA

```http
GET /api/eda/monthly-sales
GET /api/eda/region-sales
GET /api/eda/category-sales
```

### Classification

```http
POST /api/predict/buy
```

### Regression

```http
POST /api/predict/sales
```

### Forecasting

```http
GET /api/forecast/monthly-sales
```

---

# 2. Swagger

Must be available:

```text
/docs
```

```text
/redoc
```

---

# 3. MVC Structure

```text
app/

├── api/
│   ├── routers/
│   ├── schemas/
│   └── services/
│
├── ml/
│   ├── classifier/
│   ├── regressor/
│   └── forecasting/
│
├── dashboard/
│   ├── gradio_app.py
│   ├── tableau_embed.py
│   └── assets/
│
├── models/
│
├── tests/
│
└── main.py
```

---

# 4. Pydantic

Create request/response schemas for all API endpoints.

---

# 5. Async/Await

Convert API routes to async.

Example:

```python
@router.post("/predict/sales")
async def predict_sales():
```

---

# 6. pytest

Required coverage:

* EDA endpoints
* Classification endpoint
* Regression endpoint
* Forecast endpoint
* Invalid payload validation

---

# 7. Tableau Dashboard Integration

Current Tableau Dashboard must be integrated into Gradio.

Developer will receive:

```text
TABLEAU_EMBED_CODE
```

or

```text
TABLEAU_PUBLIC_URL
```

---

## New Gradio Tab

Create:

```text
📊 Tableau Dashboard
```

Implementation:

```python
gr.HTML("""
<iframe
src='<TABLEAU_PUBLIC_URL>'
width='100%'
height='900'>
</iframe>
""")
```

Dashboard must load directly inside Gradio.

---

# 8. Time Series Forecasting

Add forecasting model.

Suggested:

```python
Prophet
```

or

```python
ARIMA
```

or

```python
RandomForest + Lag Features
```

---

## New Gradio Tab

```text
📈 Future Sales Forecast
```

User selects:

* Product
* Region
* Forecast Horizon

Output:

```text
Predicted Sales Next Month
Predicted Sales Next Quarter
```

Visualization:

```python
Plotly Line Chart
```

---

# 9. Docker

Docker must run:

* FastAPI
* Gradio

inside one container.

Recommended:

```text
FastAPI (8000)
Gradio (7860)
```

---

## Dockerfile

Requirements:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 7860
EXPOSE 8000
```

---

## Container Startup

Use supervisor or startup script.

Example:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
python dashboard/gradio_app.py
```

---

# 10. Final Deliverable

Dockerized application with:

✅ FastAPI API

✅ Swagger Docs

✅ MVC Structure

✅ Pydantic Validation

✅ Async Routes

✅ pytest Coverage

✅ Tableau Dashboard Embedded in Gradio

✅ RandomForestClassifier

✅ RandomForestRegressor

✅ Time Series Forecasting

✅ Single-command Docker Startup


Do not rebuild the Tableau dashboard in Plotly. Use the Tableau embed directly inside a Gradio tab.