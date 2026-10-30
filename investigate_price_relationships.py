
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "ml_ready_data.csv"

df = pd.read_csv(DATA_PATH)

target = "Price_in_Lakhs"

print("\n1. DATASET INFORMATION")
print("=" * 45)
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Unique property prices:", df[target].nunique())

print("\n2. PRICE SUMMARY")
print("=" * 45)
print(df[target].describe())

print("\n3. NUMERICAL FEATURE CORRELATIONS")
print("=" * 45)

numeric_df = df.select_dtypes(include="number")

correlations = (
    numeric_df.corr()[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print(correlations.to_string())

print("\n4. AVERAGE PRICE BY BHK")
print("=" * 45)
print(df.groupby("BHK")[target].agg(
    ["count", "mean", "std"]
).round(2))

print("\n5. AVERAGE PRICE BY PROPERTY TYPE")
print("=" * 45)
print(df.groupby("Property_Type")[target].agg(
    ["count", "mean", "std"]
).round(2))

print("\n6. AVERAGE PRICE BY CITY")
print("=" * 45)
print(df.groupby("City")[target].agg(
    ["count", "mean", "std"]
).round(2))

print("\nInvestigation completed.")
