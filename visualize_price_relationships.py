
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "ml_ready_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Chart 1: Property size vs price
plt.figure(figsize=(9, 5))
plt.scatter(
    df["Size_in_SqFt"].sample(
        n=min(3000, len(df)), random_state=42
    ),
    df.loc[
        df["Size_in_SqFt"].sample(
            n=min(3000, len(df)), random_state=42
        ).index,
        "Price_in_Lakhs"
    ],
    alpha=0.4
)
plt.xlabel("Property Size (Sq Ft)")
plt.ylabel("Price (Lakhs)")
plt.title("Property Size vs Property Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "size_vs_price.png", dpi=150)
plt.close()

# Chart 2: Average price by BHK
bhk_prices = df.groupby("BHK")["Price_in_Lakhs"].mean()

plt.figure(figsize=(8, 5))
bhk_prices.plot(kind="bar")
plt.xlabel("Number of BHK")
plt.ylabel("Average Price (Lakhs)")
plt.title("Average Property Price by BHK")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_by_bhk.png", dpi=150)
plt.close()

# Chart 3: Average price by property type
type_prices = df.groupby(
    "Property_Type"
)["Price_in_Lakhs"].mean().sort_values()

plt.figure(figsize=(9, 5))
type_prices.plot(kind="bar")
plt.xlabel("Property Type")
plt.ylabel("Average Price (Lakhs)")
plt.title("Average Property Price by Property Type")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_by_property_type.png", dpi=150)
plt.close()

print("Charts created successfully!")
print("Saved in:", OUTPUT_DIR)
print("- size_vs_price.png")
print("- price_by_bhk.png")
print("- price_by_property_type.png")
