import numpy as np

# ---------------------------------------------
# W1D3: NumPy Array Operations
# Demonstrates:
# 1. Array statistics
# 2. Boolean masking
# 3. Matrix operations
# 4. Broadcasting (Column-wise Normalization)
# ---------------------------------------------

# Display NumPy arrays with better formatting
np.set_printoptions(precision=3, suppress=True)

# Set random seed for reproducible results
np.random.seed(42)

# EXERCISE 1: Create and explore arrays

data = np.array([100, 80, 60, 40, 20, 10, 30, 50, 70, 90])

# Calculate statistics once
mean_value = data.mean()
std_value = data.std()

print("=== Array Statistics ===")
print(f"Shape      : {data.shape}")
print(f"Mean       : {mean_value:.2f}")
print(f"Std Dev    : {std_value:.2f}")
print(f"Minimum    : {data.min()}")
print(f"Maximum    : {data.max()}")

# EXERCISE 2: Boolean Masking

threshold = mean_value
above_avg = data[data > threshold]

print("\n=== Values Above Average ===")
print(above_avg)

# EXERCISE 3: Matrix Operations

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("\n=== Matrix Operations ===")

print("\nAddition:")
print(A + B)

print("\nMatrix Multiplication:")
print(A @ B)

print("\nTranspose:")
print(A.T)

# EXERCISE 4: Broadcasting (Column-wise Normalization)

X = np.random.randint(10, 100, size=(5, 3))

print("\n=== Original Matrix ===")
print(X)

# Prevent division by zero if any column has zero standard deviation
std = X.std(axis=0)
std[std == 0] = 1

X_norm = (X - X.mean(axis=0)) / std

print("\n=== Normalized Matrix ===")
print(X_norm.round(3))