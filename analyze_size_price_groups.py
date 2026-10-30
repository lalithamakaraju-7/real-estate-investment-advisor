
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "ml_ready_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Divide properties into size ranges
df["Size_Range"] = pd.cut(
    df["Size_in_SqFt"],
    bins=[0, 1000, 2000, 3000, 4000, 5000, 10000],
    labels=[
        "0-1000",
        "1001-2000",
        "2001-3000",
        "3001-4000",
        "4001-5000",
        "5001-10000"
    ],
    include_lowest=True
)

summary = df.groupby(
    "Size_Range", observed=False
).agg(
    Property_Count=("Price_in_Lakhs", "count"),
    Average_Size=("Size_in_SqFt", "mean"),
    Average_Price=("Price_in_Lakhs", "mean"),
    Median_Price=("Price_in_Lakhs", "median"),
    Price_Std=("Price_in_Lakhs", "std")
).round(2)

print("\nPROPERTY SIZE VS PRICE")
print("=" * 65)
print(summary.to_string())

output_path = OUTPUT_DIR / "size_price_group_analysis.csv"
summary.to_csv(output_path)

print("\nAnalysis saved to:", output_path)
