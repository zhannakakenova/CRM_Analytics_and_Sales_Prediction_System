from __future__ import annotations

"""
Classification pipeline (Pipeline 1): Buy / Not Buy

What this file does:
1. Loads the trained RandomForestClassifier model.
2. Takes customer + product information from terminal arguments.
3. Predicts if customer will buy (`Buy`) or not (`Not Buy`).
4. Shows prediction probability so we know confidence.
"""

import argparse

from predict_utils import predict_buy


def main() -> None:
    parser = argparse.ArgumentParser(description="Classification pipeline: predict Buy/Not Buy")
    parser.add_argument("--customer_key", type=int, required=True)
    parser.add_argument("--product_key", type=int, required=True)
    parser.add_argument("--order_quantity", type=float, default=1)
    parser.add_argument("--list_price", type=float, default=0.0)
    parser.add_argument("--country_region", type=str, default="United States")
    parser.add_argument("--state_province", type=str, default="California")
    parser.add_argument("--category", type=str, default="Bikes")
    parser.add_argument("--subcategory", type=str, default="Mountain Bikes")
    parser.add_argument("--color", type=str, default="Black")
    parser.add_argument("--channel", type=str, default="Reseller")
    parser.add_argument("--region", type=str, default="Northwest")
    args = parser.parse_args()

    row = vars(args)
    pred, prob = predict_buy(row)

    label = "Buy" if pred == 1 else "Not Buy"
    print("=== Classification Pipeline ===")
    print(f"Result: {label}")
    print(f"Probability of Buy: {prob:.3f}")


if __name__ == "__main__":
    main()
