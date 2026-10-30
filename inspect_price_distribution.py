
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "ml_ready_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
prices = df["Price_in_Lakhs"]

print("PRICE DISTRIBUTION CHECK")
print("=" * 40)
print("Number of records:", len(prices))
print("Unique prices:", prices.nunique())
print("Minimum price:", prices.min())
print("Maximum price:", prices.max())
print("Mean price:", round(prices.mean(), 2))
print("Median price:", round(prices.median(), 2))

print("\nPRICE RANGE COUNTS")
print("=" * 40)

bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
counts = pd.cut(
    prices,
    bins=bins,
    include_lowest=True
).value_counts(sort=False)

print(counts.to_string())

# Plot the price distribution
plt.figure(figsize=(9, 5))
plt.hist(prices, bins=30, edgecolor="black")
plt.xlabel("Property Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.title("Distribution of Property Prices")
plt.tight_layout()

chart_path = OUTPUT_DIR / "price_distribution.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("\nChart saved to:", chart_path)
