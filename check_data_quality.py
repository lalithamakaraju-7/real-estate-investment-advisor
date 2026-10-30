
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/india_housing_prices.csv")

print("1. INVALID NUMERICAL VALUES")

checks = {
    "BHK": df["BHK"] <= 0,
    "Size_in_SqFt": df["Size_in_SqFt"] <= 0,
    "Price_in_Lakhs": df["Price_in_Lakhs"] <= 0,
    "Price_per_SqFt": df["Price_per_SqFt"] <= 0,
    "Year_Built": df["Year_Built"] <= 0,
    "Floor_No": df["Floor_No"] < 0,
    "Total_Floors": df["Total_Floors"] <= 0,
    "Nearby_Schools": df["Nearby_Schools"] < 0,
    "Nearby_Hospitals": df["Nearby_Hospitals"] < 0,
}

for column, condition in checks.items():
    print(f"{column}: {condition.sum()} invalid rows")

print("\n2. FLOOR CONSISTENCY")
print(
    "Floor_No greater than Total_Floors:",
    (df["Floor_No"] > df["Total_Floors"]).sum()
)

print("\n3. PRICE PER SQ FT CONSISTENCY")
calculated_price = (
    df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
)

difference = (df["Price_per_SqFt"] - calculated_price).abs()

print("Average absolute difference:", round(difference.mean(), 2))
print("Maximum absolute difference:", round(difference.max(), 2))

print("\n4. CATEGORICAL VALUES")
categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts().head(10))

print("\nData quality check completed.")

print("\n5. INSPECT SUSPICIOUS RECORDS")

# Check rows with inconsistent floor numbers
floor_issues = df[df["Floor_No"] > df["Total_Floors"]]

print("\nExamples of floor issues:")
print(
    floor_issues[
        ["ID", "City", "Floor_No", "Total_Floors"]
    ].head(10).to_string(index=False)
)

# Check whether property age matches year built
print("\nProperty age consistency:")
print(
    "Age mismatch count:",
    (df["Age_of_Property"] != (2026 - df["Year_Built"])).sum()
)

# Check the actual price-per-square-foot calculation
print("\nPrice calculation examples:")
df["Calculated_Price_per_SqFt"] = (
    df["Price_in_Lakhs"] / df["Size_in_SqFt"]
)


print(
    df[
        [
            "ID",
            "Size_in_SqFt",
            "Price_in_Lakhs",
            "Price_per_SqFt",
            "Calculated_Price_per_SqFt",
        ]
    ].head(10).round(2).to_string(index=False)
)

# Inspect unusual price-per-square-foot values
print("\nPrice per sq ft distribution:")
print(df["Price_per_SqFt"].describe().round(2))
print("\n--- PROPERTY AGE CHECK ---")

print("Year Built:")
print(df["Year_Built"].describe())

print("\nAge of Property:")
print(df["Age_of_Property"].describe())

print("\nSample age records:")
print(
    df[["ID", "Year_Built", "Age_of_Property"]]
    .head(10)
    .to_string(index=False)
)

print("\n--- FLOOR CHECK ---")

print(
    df[["ID", "Floor_No", "Total_Floors"]]
    .head(15)
    .to_string(index=False)
)

print("\n--- PRICE CHECK ---")

calculated_price = (
    df["Price_in_Lakhs"] / df["Size_in_SqFt"]
)

difference = (
    df["Price_per_SqFt"] - calculated_price
).abs()

print("Average absolute difference:", round(difference.mean(), 6))
print("Maximum absolute difference:", round(difference.max(), 6))
