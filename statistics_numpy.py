import numpy as np
<<<<<<< HEAD
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
=======

# Load CSV data (skip header)
data = np.loadtxt("student_scores.csv", delimiter=",", skiprows=1)

# Mean
print("Mean:", np.mean(data, axis=0))

# Standard Deviation
print("Standard Deviation:", np.std(data, axis=0))

# Correlation Matrix
print("Correlation Matrix:")
print(np.corrcoef(data, rowvar=False))
>>>>>>> 3486750 (feat(numpy): add statistics using CSV dataset)
