import numpy as np
import pandas as pd

# Read CSV file
data = pd.read_csv("student_scores.csv")

print("Dataset:")
print(data)

# Convert DataFrame to NumPy array
arr = data.to_numpy()

# Mean
print("\nMean:")
print(np.mean(arr, axis=0))

# Standard Deviation
print("\nStandard Deviation:")
print(np.std(arr, axis=0))

# Correlation
print("\nCorrelation Matrix:")
print(np.corrcoef(arr, rowvar=False))