# Project Plan (Simple)

## Goal
Create a CRM mini-project with:
- data cleaning,
- EDA (charts + insights),
- classification (`Buy/Not Buy`) using `RandomForestClassifier`,
- regression (sales amount) using `RandomForestRegressor`,
- Gradio dashboard for final presentation.

## Files
- `src/common.py` -> load + merge + clean data
- `src/data_exploration.py` -> EDA and charts
- `src/train_models.py` -> train both ML models
- `src/predict_classification.py` -> classification pipeline script
- `src/predict_regression.py` -> regression pipeline script
- `src/predict_utils.py` -> shared prediction helper functions
- `src/dashboard_gradio.py` -> Gradio dashboard
- `README_STUDENT.md` -> beginner guide

## Work Steps
1. Run EDA script and generate plots.
2. Train classification and regression models.
3. Run prediction script to test one sample.
4. Launch Gradio to view results and demo predictions.

## Expected Outputs
- `data/processed/merged_sales_data.csv`
- `outputs/eda/*.png`
- `outputs/eda/summary.json`
- `models/random_forest_classifier.joblib`
- `models/random_forest_regressor.joblib`
