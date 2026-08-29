"""
W3D3: Decision Trees

- Load the Iris dataset
- Split data into training and testing sets
- Train Decision Trees using Gini impurity and entropy
- Evaluate and compare both models
- Visualize the Gini Decision Tree
"""

import os

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ---------------------------------------------------------
# 1. Load the Iris dataset
# ---------------------------------------------------------
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)


# ---------------------------------------------------------
# 2. Split data into training and testing sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------------------------------
# 3. Train Decision Tree using Gini impurity
# ---------------------------------------------------------
decision_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

decision_tree.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. Evaluate the Gini Decision Tree
# ---------------------------------------------------------
train_accuracy = decision_tree.score(X_train, y_train)
test_accuracy = decision_tree.score(X_test, y_test)

print("\nDecision Tree - Gini Impurity")
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
print("Tree depth:", decision_tree.get_depth())
print("Number of leaves:", decision_tree.get_n_leaves())


# ---------------------------------------------------------
# 5. Train Decision Tree using Entropy
# ---------------------------------------------------------
decision_tree_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

decision_tree_entropy.fit(X_train, y_train)


# ---------------------------------------------------------
# 6. Evaluate the Entropy-based Decision Tree
# ---------------------------------------------------------
entropy_train_accuracy = decision_tree_entropy.score(X_train, y_train)
entropy_test_accuracy = decision_tree_entropy.score(X_test, y_test)

print("\nDecision Tree - Entropy / Information Gain")
print("Training accuracy:", entropy_train_accuracy)
print("Testing accuracy:", entropy_test_accuracy)
print("Tree depth:", decision_tree_entropy.get_depth())
print("Number of leaves:", decision_tree_entropy.get_n_leaves())


# ---------------------------------------------------------
# 7. Compare Gini and Entropy
# ---------------------------------------------------------
print("\nCriterion Comparison")
print("--------------------------------")
print(f"Gini    - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
print(
    f"Entropy - Train: {entropy_train_accuracy:.4f}, "
    f"Test: {entropy_test_accuracy:.4f}"
)


# ---------------------------------------------------------
# 8. Create output evidence directory
# ---------------------------------------------------------
evidence_dir = os.path.join("output_evidence", "w3d3")
os.makedirs(evidence_dir, exist_ok=True)


# ---------------------------------------------------------
# 9. Visualize the Gini Decision Tree
# ---------------------------------------------------------
plt.figure(figsize=(16, 10))

plot_tree(
    decision_tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Gini Impurity")
plt.tight_layout()

tree_path = os.path.join(
    evidence_dir,
    "decision_tree_gini.png"
)

plt.savefig(
    tree_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nDecision tree visualization saved to:", tree_path)


# ---------------------------------------------------------
# 10. Validation tests
# ---------------------------------------------------------
assert X_train.shape[0] == 120
assert X_test.shape[0] == 30

assert 0 <= train_accuracy <= 1
assert 0 <= test_accuracy <= 1

assert 0 <= entropy_train_accuracy <= 1
assert 0 <= entropy_test_accuracy <= 1

assert os.path.exists(tree_path)

assert decision_tree.get_depth() > 0
assert decision_tree.get_n_leaves() > 0

assert decision_tree_entropy.get_depth() > 0
assert decision_tree_entropy.get_n_leaves() > 0

print("\nAll validation tests passed.")