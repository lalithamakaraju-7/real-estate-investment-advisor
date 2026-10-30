
import pandas as pd
from pathlib import Path

# Load the original dataset
input_path = Path("data/india_housing_prices.csv")
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Remove duplicate rows, if any
df = df.drop_duplicates().copy()

# Correct invalid price-per-square-foot values.
# Price is stored in lakhs, so the result is lakhs per sq. ft.
invalid_price = df["Price_per_SqFt"] <= 0

df.loc[invalid_price, "Price_per_SqFt"] = (
    df.loc[invalid_price, "Price_in_Lakhs"]
    / df.loc[invalid_price, "Size_in_SqFt"]
)

# Flag suspicious floor records without deleting them
df["Floor_Data_Issue"] = (
    df["Floor_No"] > df["Total_Floors"]
)

# Record the reference year implied by the dataset's age values
df["Age_Reference_Year"] = (
    df["Year_Built"] + df["Age_of_Property"]
)

# Save the cleaned dataset separately
output_path = Path("data/india_housing_prices_cleaned.csv")
df.to_csv(output_path, index=False)

print("Cleaned dataset shape:", df.shape)
print("Duplicate rows remaining:", df.duplicated().sum())
print("Invalid price-per-sq-ft values remaining:",
      (df["Price_per_SqFt"] <= 0).sum())
print("Rows flagged for floor issues:",
      df["Floor_Data_Issue"].sum())
print("Age reference years:")
print(df["Age_Reference_Year"].value_counts().head())
print("Saved cleaned dataset to:", output_path)
