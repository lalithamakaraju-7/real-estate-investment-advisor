
import pandas as pd

# Load the dataset
df = pd.read_csv("data/india_housing_prices.csv")

# 1. Dataset dimensions
print("Dataset Shape:", df.shape)

# 2. Column names and data types
print("\nDataset Information:")
df.info()

# 3. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Duplicate records
print("\nDuplicate Rows:", df.duplicated().sum())

# 5. Statistical summary
print("\nNumerical Summary:")
print(df.describe())

# 6. First five records
print("\nFirst Five Records:")
print(df.head())
