from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.api.services.dataset_service import build_merged_dataset, clean_dataset
from app.core import MODELS_DIR
from app.ml.classifier.service import CLASSIFIER_FEATURES
from app.ml.regressor.service import REGRESSOR_FEATURES

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_classifier_dataset(df: pd.DataFrame) -> pd.DataFrame:
    positive = (
        df.groupby(["customer_key", "product_key"], as_index=False)
        .agg(
            sales_amount=("sales_amount", "sum"),
            country_region=("country_region", "first"),
            state_province=("state_province", "first"),
            category=("category", "first"),
            subcategory=("subcategory", "first"),
            color=("color", "first"),
            list_price=("list_price", "mean"),
            channel=("channel", "first"),
            region=("region", "first"),
        )
    )
    positive["buy_label"] = 1

    rng = np.random.default_rng(42)
    customers = df["customer_key"].dropna().unique()
    products = df["product_key"].dropna().unique()
    purchased = set(zip(positive["customer_key"], positive["product_key"]))

    negative_rows = []
    target_negative_count = min(len(positive), 50000)
    while len(negative_rows) < target_negative_count:
        customer_key = int(rng.choice(customers))
        product_key = int(rng.choice(products))
        if (customer_key, product_key) not in purchased:
            negative_rows.append((customer_key, product_key))

    negative = pd.DataFrame(negative_rows, columns=["customer_key", "product_key"]).drop_duplicates()
    customer_info = df.groupby("customer_key", as_index=False).agg(
        country_region=("country_region", "first"),
        state_province=("state_province", "first"),
        region=("region", "first"),
        channel=("channel", "first"),
    )
    product_info = df.groupby("product_key", as_index=False).agg(
        category=("category", "first"),
        subcategory=("subcategory", "first"),
        color=("color", "first"),
        list_price=("list_price", "mean"),
    )

    negative = negative.merge(customer_info, on="customer_key", how="left")
    negative = negative.merge(product_info, on="product_key", how="left")
    negative["sales_amount"] = 0.0
    negative["buy_label"] = 0

    return pd.concat([positive, negative], ignore_index=True)


def train_classifier(df: pd.DataFrame) -> None:
    classifier_df = _build_classifier_dataset(df)
    X = classifier_df[CLASSIFIER_FEATURES]
    y = classifier_df["buy_label"]

    categorical_columns = [
        "country_region",
        "state_province",
        "category",
        "subcategory",
        "color",
        "channel",
        "region",
    ]
    numeric_columns = ["customer_key", "product_key", "list_price"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== Classification Results ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, digits=4))
    joblib.dump(model, MODELS_DIR / "random_forest_classifier.joblib", compress=3)


def train_regressor(df: pd.DataFrame) -> None:
    X = df[REGRESSOR_FEATURES]
    y = df["sales_amount"]

    categorical_columns = [
        "country_region",
        "state_province",
        "category",
        "subcategory",
        "color",
        "channel",
        "region",
        "month",
        "fiscal_quarter",
    ]
    numeric_columns = [
        "customer_key",
        "product_key",
        "order_quantity",
        "unit_price",
        "unit_price_discount_pct",
        "list_price",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("reg", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== Regression Results ===")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"R2:  {r2_score(y_test, predictions):.4f}")
    joblib.dump(model, MODELS_DIR / "random_forest_regressor.joblib", compress=3)


def main() -> None:
    df = clean_dataset(build_merged_dataset())
    train_classifier(df)
    train_regressor(df)
    print("\nModels saved in models/")


if __name__ == "__main__":
    main()
