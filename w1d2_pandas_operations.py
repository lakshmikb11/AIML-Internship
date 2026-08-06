import pandas as pd
import os

# Load Indian population dataset into Pandas DataFrame
df = pd.read_csv("data/indian population.csv")

# Print dataset shape (number of rows and columns)
print("Dataset Shape:")
print(df.shape)

# Print column names and their data types
print("\nData Types:")
print(df.dtypes)

# Print first 10 rows of dataset
print("\nFirst 10 Rows:")
print(df.head(10))
# Print all column names
print("\nColumn Names:")
print(df.columns.tolist())
# Filter: Select areas where total population is greater than 1 crore
filtered_df = df[df["TOT_P"] > 10000000]

print("\nFiltered Data (Population > 1 crore):")
print(filtered_df)
# Groupby: Calculate total population for each state
grouped_df = df.groupby("State")["TOT_P"].sum()

print("\nTotal Population by State:")
print(grouped_df)
# Create another DataFrame for merging
state_info = pd.DataFrame({
    "State": df["State"].unique(),
    "State_Name": ["State_" + str(i) for i in df["State"].unique()]
})

# Merge original dataset with state information
merged_df = pd.merge(df, state_info, on="State")

print("\nMerged Data:")
print(merged_df.head())
# Pivot table: Summarize population based on State and TRU
pivot_df = pd.pivot_table(
    df,
    values="TOT_P",
    index="State",
    columns="TRU",
    aggfunc="sum"
)

print("\nPivot Table:")
print(pivot_df)
# Export cleaned DataFrame to CSV file
df.to_csv("india_population_cleaned.csv", index=False)

print("\nCSV file exported successfully!")

# Export cleaned DataFrame to Parquet file
df.to_parquet("india_population_cleaned.parquet", index=False)

print("Parquet file exported successfully!")

import os

# Compare CSV and Parquet file sizes
csv_size = os.path.getsize("india_population_cleaned.csv")
parquet_size = os.path.getsize("india_population_cleaned.parquet")

print("\nFile Size Comparison:")
print("CSV Size:", csv_size, "bytes")
print("Parquet Size:", parquet_size, "bytes")