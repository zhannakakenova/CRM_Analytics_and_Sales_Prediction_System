# Machine Learning Methodology

This project separates three different machine learning tasks. They answer different business questions and must not be interpreted as the same type of prediction.

| Task | Question | Output | Method |
| --- | --- | --- | --- |
| Classification | Will this customer buy this product? | `Buy` or `Not Buy` with probability | Random Forest Classifier |
| Regression | What sales amount is expected for this transaction? | Predicted sales amount | Random Forest Regressor |
| Forecasting | How much will be sold in future months? | Future monthly sales trend | Time-series features with Random Forest Regressor |

## Classification: Buy or Not Buy

### Why the Not Buy Class Must Be Created

The AdventureWorks sales data contains completed purchases. Every customer-product pair in the sales table is therefore a known **Buy** example.

The source data does not contain explicit records saying that a customer considered a product and decided not to buy it. A supervised classifier requires both positive and negative classes, so the project creates a synthetic **Not Buy** class during training.

### How Synthetic Not Buy Examples Are Created

1. Group historical sales by `customer_key` and `product_key`.
2. Label every observed customer-product pair as `Buy`.
3. Randomly sample customer-product pairs that do not appear in the observed purchases.
4. Label those unobserved pairs as `Not Buy`.
5. Join known customer attributes, including the customer's observed channel, and known product attributes onto the synthetic pairs.
6. Train a balanced Random Forest Classifier using observed Buy pairs and sampled Not Buy pairs.

The random generator uses seed `42` so the sampled negative class is reproducible.

### Important Limitation

An unobserved customer-product pair does not prove that the customer rejected the product. The customer may never have seen it. Therefore, the synthetic Not Buy class is an approximation suitable for an educational project, not a true record of customer intent.

### Why Order Quantity Is Excluded

Order quantity is only known after or during a purchase. Using it to predict whether a customer will buy would leak information from the outcome into the model.

Earlier versions also assigned quantity `0` to synthetic Not Buy rows, which made quantity an almost direct indicator of the class. The classifier now excludes order quantity entirely.

## Regression: Predicted Sales Amount

The Sales Prediction tab uses the saved `RandomForestRegressor` from:

```text
models/random_forest_regressor.joblib
```

The service sends transaction, customer, product, location, channel, month, and fiscal-quarter features into the trained model. It returns the model's estimated sales amount.

The prediction is no longer calculated directly using:

```text
quantity × unit price × (1 - discount)
```

Although those values remain useful model features, the output comes from `RandomForestRegressor.predict()`.

### Regression Features

- Customer key
- Product key
- Order quantity
- Unit price
- Discount percentage
- List price
- Customer location
- Product category, subcategory, and color
- Sales channel and territory
- Month and fiscal quarter

### Important Limitation

Historical sales amount is strongly related to quantity, price, and discount. The model may learn a relationship close to the normal sales formula. It is still a trained regression model, but evaluation metrics such as MAE and R² should be used to judge its quality.

## Forecasting: Future Monthly Sales

Forecasting predicts a sequence of future monthly totals rather than one transaction.

The forecasting service:

1. Aggregates historical sales by month.
2. Creates lag features from the previous three months.
3. Adds month-number and trend features.
4. Trains a Random Forest Regressor on the monthly series.
5. Predicts future months recursively.

Forecasting can optionally filter the history by product or sales territory.

## Model Evaluation

Running the training command prints classification and regression metrics:

```bash
python3 -m app.ml.training
```

The classifier reports accuracy and a classification report. The regressor reports Mean Absolute Error (MAE) and R².

Metrics should always be interpreted together with the limitations of the synthetic Not Buy class and the historical dataset.
