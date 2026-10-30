
import pandas as pd

df = pd.read_csv("data/india_housing_prices_cleaned.csv")

print("DATASET SUITABILITY REPORT")
print("Dataset shape:", df.shape)

# Check price variation across important features
columns = [
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Furnished_Status",
    "Owner_Type",
    "Availability_Status",
]

print("\nUNIQUE VALUES")
for col in columns:
    print(f"{col}: {df[col].nunique()}")

# Check whether locality is related to property price
print("\nPRICE BY LOCALITY")
locality_stats = df.groupby("Locality")["Price_in_Lakhs"].agg(
    ["count", "mean", "std"]
)

print("\n10 Localities with highest average prices:")
print(locality_stats.sort_values("mean", ascending=False).head(10).round(2))

print("\n10 Localities with lowest average prices:")
print(locality_stats.sort_values("mean", ascending=True).head(10).round(2))

# Compare overall price variation
print("\nPRICE VARIATION")
print("Overall price standard deviation:",
      round(df["Price_in_Lakhs"].std(), 2))

print("Average locality price standard deviation:",
      round(locality_stats["mean"].std(), 2))

print("\nDATASET SUITABILITY CHECK COMPLETED")
