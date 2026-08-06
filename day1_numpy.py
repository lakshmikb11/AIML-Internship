import numpy as np

# 1D Array
arr1 = np.array([10, 20, 30, 40])
print("1D Array:")
print(arr1)
print("Shape:", arr1.shape)

# 2D Array
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
print("\n2D Array:")
print(arr2)
print("Shape:", arr2.shape)

# 3D Array
arr3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("\n3D Array:")
print(arr3)
print("Shape:", arr3.shape)

# Broadcasting
print("\nBroadcasting:")
print(arr1 + 5)

# Vectorized Operations
print("\nVectorized Multiplication:")
print(arr1 * 2)

# Matrix Multiplication
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("\nMatrix Multiplication:")
print(np.matmul(A, B))