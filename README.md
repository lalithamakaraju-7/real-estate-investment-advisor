# Real Estate Investment Advisor

## Project Overview
The Real Estate Investment Advisor is a Machine Learning project that analyzes property information and estimates property prices. It also provides a hypothetical five-year future value scenario based on a user-selected annual growth rate.

## Objectives
- Analyze real estate property data.
- Predict property prices using Machine Learning.
- Explore property features and their relationship with price.
- Display predictions through a Streamlit web application.
- Track model experiments and evaluation metrics using MLflow.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- MLflow
- Joblib

## Machine Learning Model
- Algorithm: ExtraTreesRegressor
- Task: Property price prediction
- Evaluation metrics: MAE, RMSE, and R²
- Baseline comparison: Mean prediction baseline

## Model Limitations
The current model did not outperform the mean baseline on the evaluation dataset. Its predictions should therefore be treated as experimental, not reliable property valuations.

The five-year future value is a hypothetical scenario based on an assumed annual growth rate, not a market forecast.

## Application
Run the application using:

```bash
streamlit run app.py
```

## Experiment Tracking
MLflow is used to record model parameters and evaluation metrics.

## Disclaimer
This project is for educational purposes only. It does not provide professional real estate investment or financial advice.