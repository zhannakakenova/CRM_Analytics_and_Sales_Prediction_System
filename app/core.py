from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_PATH = PROCESSED_DATA_DIR / "merged_sales_data.csv"
EDA_DIR = PROJECT_ROOT / "outputs" / "eda"
MODELS_DIR = PROJECT_ROOT / "models"
