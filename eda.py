import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Load cleaned dataset
df = pd.read_csv("data/india_housing_prices_cleaned.csv")

# Create folder for charts
Path("outputs").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Dataset overview
print("DATASET OVERVIEW")
print("Dataset shape:", df.shape)

print("\nNumerical summary:")
print(df.describe().round(2))

# 2. Property price distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Price_in_Lakhs", bins=50, kde=True)
plt.title("Distribution of Property Prices")
plt.xlabel("Property Price (Lakhs)")
plt.tight_layout()
plt.savefig("outputs/price_distribution.png")


# 3. Average price by property type
type_prices = (
    df.groupby("Property_Type")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAVERAGE PRICE BY PROPERTY TYPE")
print(type_prices.round(2))

plt.figure(figsize=(9, 5))
sns.barplot(x=type_prices.index, y=type_prices.values)
plt.title("Average Property Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/average_price_by_type.png")


# 4. Top 10 cities by average price
city_prices = (
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP 10 CITIES BY AVERAGE PROPERTY PRICE")
print(city_prices.round(2))

plt.figure(figsize=(10, 6))
sns.barplot(x=city_prices.values, y=city_prices.index)
plt.title("Top 10 Cities by Average Property Price")
plt.xlabel("Average Price (Lakhs)")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("outputs/top_10_cities_by_price.png")


print("\nEDA completed successfully!")
print("Charts saved in the outputs folder.")
