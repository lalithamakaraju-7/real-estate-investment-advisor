
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.dummy import DummyRegressor

# Project paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "ml_ready_data.csv"
MODEL_PATH = BASE_DIR / "real_estate_price_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load data and trained model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

target = "Price_in_Lakhs"

features = [
    "City",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status"
]

X = df[features]
y = df[target]

# Use the same split settings as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Evaluate trained model
model_predictions = model.predict(X_test)

# Baseline: always predict the training-set average
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted))
    }

results = {
    "Trained_Model": calculate_metrics(y_test, model_predictions),
    "Mean_Baseline": calculate_metrics(y_test, baseline_predictions)
}

print("\nMODEL EVALUATION RESULTS")
print("=" * 40)

for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Save results
output_path = OUTPUT_DIR / "model_evaluation.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print(f"\nResults saved to: {output_path}")
