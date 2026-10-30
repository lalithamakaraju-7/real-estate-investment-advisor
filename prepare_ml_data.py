
import pandas as pd
from pathlib import Path

# 1. Locate the project folder
BASE_DIR = Path(__file__).resolve().parent

# 2. Load the cleaned dataset
data_path = BASE_DIR / "data" / "india_housing_prices_cleaned.csv"
df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)

# 3. Remove rows with missing target prices
df = df.dropna(subset=["Price_in_Lakhs"]).copy()

# 4. Select useful features for price prediction
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
    "Availability_Status",
]

target = "Price_in_Lakhs"

# 5. Keep only the selected features and target
ml_df = df[features + [target]].copy()

# 6. Save the prepared dataset
output_path = BASE_DIR / "data" / "ml_ready_data.csv"
ml_df.to_csv(output_path, index=False)

print("Prepared dataset shape:", ml_df.shape)
print("Missing values:", int(ml_df.isnull().sum().sum()))
print("Saved to:", output_path)
print("\nFirst 5 rows:")
print(ml_df.head())
