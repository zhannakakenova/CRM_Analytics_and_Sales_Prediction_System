from __future__ import annotations

from pathlib import Path

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

from common import build_merged_dataset, clean_dataset

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_classifier_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Positive samples: existing purchases.
    pos = (
        df.groupby(["customer_key", "product_key"], as_index=False)
        .agg(
            order_quantity=("order_quantity", "sum"),
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
    pos["buy_label"] = 1

    # Negative samples: random customer-product pairs not found in purchases.
    rng = np.random.default_rng(42)
    customers = df["customer_key"].dropna().unique()
    products = df["product_key"].dropna().unique()
    purchased = set(zip(pos["customer_key"], pos["product_key"]))

    target_neg_count = min(len(pos), 50000)
    neg_rows = []
    while len(neg_rows) < target_neg_count:
        c = int(rng.choice(customers))
        p = int(rng.choice(products))
        if (c, p) in purchased:
            continue
        neg_rows.append((c, p))

    neg = pd.DataFrame(neg_rows, columns=["customer_key", "product_key"]).drop_duplicates()

    customer_info = df.groupby("customer_key", as_index=False).agg(
        country_region=("country_region", "first"),
        state_province=("state_province", "first"),
        region=("region", "first"),
    )
    product_info = df.groupby("product_key", as_index=False).agg(
        category=("category", "first"),
        subcategory=("subcategory", "first"),
        color=("color", "first"),
        list_price=("list_price", "mean"),
    )

    neg = neg.merge(customer_info, on="customer_key", how="left")
    neg = neg.merge(product_info, on="product_key", how="left")
    neg["order_quantity"] = 0
    neg["sales_amount"] = 0.0
    neg["channel"] = "Unknown"
    neg["buy_label"] = 0

    cls_df = pd.concat([pos, neg], ignore_index=True)
    return cls_df


def train_classifier(df: pd.DataFrame) -> None:
    cls_df = _build_classifier_dataset(df)
    features = [
        "customer_key",
        "product_key",
        "order_quantity",
        "country_region",
        "state_province",
        "category",
        "subcategory",
        "color",
        "list_price",
        "channel",
        "region",
    ]
    target = "buy_label"

    X = cls_df[features]
    y = cls_df[target]

    cat_cols = ["country_region", "state_province", "category", "subcategory", "color", "channel", "region"]
    num_cols = ["customer_key", "product_key", "order_quantity", "list_price"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n=== Classification Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4))

    joblib.dump(model, MODELS_DIR / "random_forest_classifier.joblib")


def train_regressor(df: pd.DataFrame) -> None:
    reg_df = df.copy()

    features = [
        "customer_key",
        "product_key",
        "order_quantity",
        "unit_price",
        "unit_price_discount_pct",
        "list_price",
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
    target = "sales_amount"

    X = reg_df[features]
    y = reg_df[target]

    cat_cols = [
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
    num_cols = ["customer_key", "product_key", "order_quantity", "unit_price", "unit_price_discount_pct", "list_price"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
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

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n=== Regression Results ===")
    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")

    joblib.dump(model, MODELS_DIR / "random_forest_regressor.joblib")


def main() -> None:
    df = clean_dataset(build_merged_dataset())
    train_classifier(df)
    train_regressor(df)
    print("\nModels saved in models/")


if __name__ == "__main__":
    main()
