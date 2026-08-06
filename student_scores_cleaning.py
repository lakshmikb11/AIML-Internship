import pandas as pd

# Load CSV file
df = pd.read_csv("student_scores.csv")

print("Cleaned Dataset:")
print(df)

print("\nTotal rows:", len(df))