
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Find the project folder
BASE_DIR = Path(__file__).resolve().parent

# 2. Load prepared data
data_path = BASE_DIR / "data" / "ml_ready_data.csv"
df = pd.read_csv(data_path)

print("Dataset loaded:", df.shape)

# 3. Separate features (X) and target (y)
X = df.drop(columns=["Price_in_Lakhs"])
y = df["Price_in_Lakhs"]

# 4. Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("Categorical columns:", len(categorical_columns))
print("Numerical columns:", len(numerical_columns))

# 5. Prepare numerical and categorical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 6. Create a faster tree-based model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ExtraTreesRegressor(
        n_estimators=10,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 8. Train the model
print("Training model. Please wait...", flush=True)
model.fit(X_train, y_train)
print("Training completed!", flush=True)


# 9. Predict prices for test data
predictions = model.predict(X_test)

# 10. Create a simple baseline prediction
# The baseline always predicts the average training-set price
baseline_predictions = [y_train.mean()] * len(y_test)

# 11. Evaluate the ML model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# 12. Evaluate the baseline
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\nMODEL EVALUATION")
print(f"MAE: {mae:.2f} lakhs")
print(f"RMSE: {rmse:.2f} lakhs")
print(f"R-squared (R2): {r2:.4f}")

print("\nBASELINE EVALUATION")
print(f"MAE: {baseline_mae:.2f} lakhs")
print(f"RMSE: {baseline_rmse:.2f} lakhs")
print(f"R-squared (R2): {baseline_r2:.4f}")

print("\nCOMPARISON")
if mae < baseline_mae:
    print("The ML model has lower MAE than the baseline.")
else:
    print("The ML model does not improve on the baseline MAE.")

# 13. Save the trained model
model_path = BASE_DIR / "real_estate_price_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
