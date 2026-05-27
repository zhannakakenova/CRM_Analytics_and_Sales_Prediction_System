from __future__ import annotations

"""
Regression pipeline (Pipeline 2): Sales Amount Prediction

What this file does:
1. Loads the trained RandomForestRegressor model.
2. Takes product + region + customer information from terminal arguments.
3. Predicts numeric `sales_amount`.
4. Prints expected sales value for planning and analysis.
"""

import argparse

from predict_utils import predict_sales


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression pipeline: predict sales amount")
    parser.add_argument("--customer_key", type=int, required=True)
    parser.add_argument("--product_key", type=int, required=True)
    parser.add_argument("--order_quantity", type=float, default=1)
    parser.add_argument("--unit_price", type=float, default=0.0)
    parser.add_argument("--unit_price_discount_pct", type=float, default=0.0)
    parser.add_argument("--list_price", type=float, default=0.0)
    parser.add_argument("--country_region", type=str, default="United States")
    parser.add_argument("--state_province", type=str, default="California")
    parser.add_argument("--category", type=str, default="Bikes")
    parser.add_argument("--subcategory", type=str, default="Mountain Bikes")
    parser.add_argument("--color", type=str, default="Black")
    parser.add_argument("--channel", type=str, default="Reseller")
    parser.add_argument("--region", type=str, default="Northwest")
    parser.add_argument("--month", type=str, default="January")
    parser.add_argument("--fiscal_quarter", type=str, default="Q1")
    args = parser.parse_args()

    row = vars(args)
    prediction = predict_sales(row)

    print("=== Regression Pipeline ===")
    print(f"Predicted sales_amount: {prediction:.2f}")


if __name__ == "__main__":
    main()
