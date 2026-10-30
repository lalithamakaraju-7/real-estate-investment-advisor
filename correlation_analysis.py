
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned data
df = pd.read_csv("data/india_housing_prices_cleaned.csv")

# Select useful numeric features
numeric_columns = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
]

correlation = df[numeric_columns].corr()

print("CORRELATION WITH PROPERTY PRICE")
print(
    correlation["Price_in_Lakhs"]
    .sort_values(ascending=False)
    .round(3)
)

# Plot correlation heatmap
Path("outputs").mkdir(exist_ok=True)

plt.figure(figsize=(11, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

print("\nCorrelation heatmap saved to outputs/correlation_heatmap.png")

