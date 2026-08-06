import numpy as np

# Load CSV data (skip header)
data = np.loadtxt("student_scores.csv", delimiter=",", skiprows=1)

# Mean
print("Mean:", np.mean(data, axis=0))

# Standard Deviation
print("Standard Deviation:", np.std(data, axis=0))

# Correlation Matrix
print("Correlation Matrix:")
print(np.corrcoef(data, rowvar=False))