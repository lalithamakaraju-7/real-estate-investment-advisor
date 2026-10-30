
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/india_housing_prices_cleaned.csv")

print("DATASET CHECK")
print("Rows and columns:", df.shape)

print("\nPROPERTY PRICE SUMMARY")
print(df["Price_in_Lakhs"].describe().round(2))

print("\nPROPERTY SIZE SUMMARY")
print(df["Size_in_SqFt"].describe().round(2))

print("\nPRICE BY BHK")
print(
    df.groupby("BHK")["Price_in_Lakhs"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

print("\nPRICE BY PROPERTY TYPE")
print(
    df.groupby("Property_Type")["Price_in_Lakhs"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

print("\nPRICE BY CITY: TOP 10")
print(
    df.groupby("City")["Price_in_Lakhs"]
    .agg(["count", "mean", "std"])
    .sort_values("mean", ascending=False)
    .head(10)
    .round(2)
)

print("\nPRICE-SIZE RELATIONSHIP")

print(
    "Price correlation with size:",
    round(df["Price_in_Lakhs"].corr(df["Size_in_SqFt"]), 4)
)

print("\nPRICE PER SQ FT CHECK")

df["Calculated_Price_Per_SqFt"] = (
    df["Price_in_Lakhs"] / df["Size_in_SqFt"]
)

print(
    df[
        ["Price_in_Lakhs", "Size_in_SqFt",
         "Price_per_SqFt", "Calculated_Price_Per_SqFt"]
    ].head(10).round(3)
)

print("\nCORRELATION WITH PROPERTY PRICE")
print(
    df[["Price_in_Lakhs", "Size_in_SqFt", "BHK"]]
    .corr()["Price_in_Lakhs"]
    .round(4)
)

print("\nAVERAGE PRICE BY SIZE GROUP")

df["Size_Group"] = pd.cut(
    df["Size_in_SqFt"],
    bins=[0, 1000, 2000, 3000, 4000, 5000],
    labels=[
        "Up to 1000",
        "1001-2000",
        "2001-3000",
        "3001-4000",
        "4001-5000"
    ],
    include_lowest=True
)

print(
    df.groupby("Size_Group", observed=False)["Price_in_Lakhs"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nPRICE VARIATION BY PROPERTY TYPE")
print(
    df.groupby("Property_Type")["Price_in_Lakhs"]
    .std()
    .round(2)
)

print("\nPRICE VARIATION BY CITY")
city_stats = df.groupby("City")["Price_in_Lakhs"].agg(
    ["mean", "std", "count"]
)

print(city_stats.sort_values("mean", ascending=False).head(10).round(2))

print("\nPRICE GENERATION INVESTIGATION")

# Check whether price changes with property size
print("\nPrice correlation with size:")
print(df["Price_in_Lakhs"].corr(df["Size_in_SqFt"]))

# Check whether price changes with BHK
print("\nAverage price for each BHK:")
print(df.groupby("BHK")["Price_in_Lakhs"].mean().round(2))

# Check whether price changes with property type
print("\nAverage price for each property type:")
print(df.groupby("Property_Type")["Price_in_Lakhs"].mean().round(2))

# Check the number of unique prices
print("\nUnique property prices:")
print(df["Price_in_Lakhs"].nunique())

# Check whether price is almost uniformly distributed
print("\nPrice distribution by range:")
print(
    pd.cut(df["Price_in_Lakhs"], bins=10)
    .value_counts()
    .sort_index()
)

print("\nALL NUMERICAL FEATURES VS PRICE")

numeric_df = df.select_dtypes(include="number")

correlations = (
    numeric_df.corr(numeric_only=True)["Price_in_Lakhs"]
    .drop("Price_in_Lakhs")
    .sort_values(key=abs, ascending=False)
)

print(correlations.round(4))
