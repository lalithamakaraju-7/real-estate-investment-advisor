# Real Estate Investment Advisor
## Model Evaluation and Limitations Report

### 1. Project objective

The project explores machine learning for estimating residential
property prices and illustrating hypothetical five-year value scenarios.

### 2. Dataset summary

- Number of records: 250,000
- Number of ML-ready columns: 18
- Missing values in the ML-ready dataset: 0
- Target: Price_in_Lakhs
- Minimum recorded price: 10 lakhs
- Maximum recorded price: 500 lakhs
- Average recorded price: 254.59 lakhs

### 3. Model evaluation

The model was evaluated using an 80/20 train-test split.

| Metric | Trained model | Mean baseline |
|---|---:|---:|
| MAE (lakhs) | 122.4344 | 122.3178 |
| RMSE (lakhs) | 141.3689 | 141.1932 |
| R-squared | -0.0025 | -0.0000 |

The trained model performs slightly worse than the mean baseline
on this evaluation split.

### 4. Data investigation findings

- Price has almost zero linear correlation with property size.
- Average prices are similar across BHK groups.
- Average prices are similar across property types.
- Average prices are similar across the cities examined.
- Price ranges contain broadly similar numbers of records.

These findings raise concerns about whether the dataset represents
realistic property-market relationships.

### 5. Limitations

1. The current model has not demonstrated reliable price prediction.
2. The observed relationships may reflect limitations in the dataset.
3. The evaluation does not establish performance on real market data.
4. No validated investment-return or profitability model is available.
5. Five-year values are scenarios based on an assumed growth rate,
   not market forecasts.

### 6. Responsible use

This project is an educational demonstration, not a professional
property valuation or investment recommendation tool.

Do not use its predictions to make actual property purchases,
financial commitments, or investment decisions.

### 7. Future improvements

- Validate the dataset against reliable property-market data.
- Investigate feature quality and target generation.
- Compare models using a consistent test split.
- Evaluate performance on an independent, realistic dataset.
- Track experiments using MLflow.
- Keep assumptions and model limitations visible in the app.
