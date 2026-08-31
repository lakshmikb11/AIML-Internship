"""
W3D4: SVM & KNN — When to Use What

Cynaris AI/ML Internship
Task: Compare Support Vector Machine (SVM) and K-Nearest Neighbors (KNN)
using the Iris dataset.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Target classes:", iris.target_names)


# ---------------------------------------------------------
# 2. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 3. Create Evidence Directory
# ---------------------------------------------------------

OUTPUT_DIR = "output_evidence/w3d4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# SVM
# =========================================================

print("\n========== SVM ==========")


# ---------------------------------------------------------
# 4. SVM Pipeline
# ---------------------------------------------------------

svm_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ]
)


# ---------------------------------------------------------
# 5. SVM Hyperparameter Tuning
# ---------------------------------------------------------

svm_param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", "auto"],
}

svm_grid = GridSearchCV(
    svm_pipeline,
    svm_param_grid,
    cv=5,
    scoring="accuracy",
)

svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_

print("Best SVM parameters:", svm_grid.best_params_)
print("Best cross-validation accuracy:", svm_grid.best_score_)


# ---------------------------------------------------------
# 6. SVM Evaluation
# ---------------------------------------------------------

svm_predictions = best_svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(
    y_test,
    svm_predictions,
    average="weighted",
    zero_division=0,
)
svm_recall = recall_score(
    y_test,
    svm_predictions,
    average="weighted",
    zero_division=0,
)
svm_f1 = f1_score(
    y_test,
    svm_predictions,
    average="weighted",
    zero_division=0,
)

print("\nSVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1-score:", svm_f1)

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))


# ---------------------------------------------------------
# 7. SVM Confusion Matrix
# ---------------------------------------------------------

svm_cm = confusion_matrix(y_test, svm_predictions)

svm_display = ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=iris.target_names,
)

svm_display.plot()
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "svm_confusion_matrix.png"),
    dpi=150,
)
plt.close()


# =========================================================
# KNN
# =========================================================

print("\n========== KNN ==========")


# ---------------------------------------------------------
# 8. KNN Pipeline
# ---------------------------------------------------------

knn_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)


# ---------------------------------------------------------
# 9. KNN Hyperparameter Tuning
# ---------------------------------------------------------

knn_param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}

knn_grid = GridSearchCV(
    knn_pipeline,
    knn_param_grid,
    cv=5,
    scoring="accuracy",
)

knn_grid.fit(X_train, y_train)

best_knn = knn_grid.best_estimator_

print("Best KNN parameters:", knn_grid.best_params_)
print("Best cross-validation accuracy:", knn_grid.best_score_)


# ---------------------------------------------------------
# 10. KNN Evaluation
# ---------------------------------------------------------

knn_predictions = best_knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(
    y_test,
    knn_predictions,
    average="weighted",
    zero_division=0,
)
knn_recall = recall_score(
    y_test,
    knn_predictions,
    average="weighted",
    zero_division=0,
)
knn_f1 = f1_score(
    y_test,
    knn_predictions,
    average="weighted",
    zero_division=0,
)

print("\nKNN Accuracy:", knn_accuracy)
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)
print("KNN F1-score:", knn_f1)

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_predictions))


# ---------------------------------------------------------
# 11. KNN Confusion Matrix
# ---------------------------------------------------------

knn_cm = confusion_matrix(y_test, knn_predictions)

knn_display = ConfusionMatrixDisplay(
    confusion_matrix=knn_cm,
    display_labels=iris.target_names,
)

knn_display.plot()
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "knn_confusion_matrix.png"),
    dpi=150,
)
plt.close()


# =========================================================
# Model Comparison
# =========================================================

print("\n========== MODEL COMPARISON ==========")


comparison = pd.DataFrame(
    {
        "Model": ["SVM", "KNN"],
        "Accuracy": [svm_accuracy, knn_accuracy],
        "Precision": [svm_precision, knn_precision],
        "Recall": [svm_recall, knn_recall],
        "F1-Score": [svm_f1, knn_f1],
    }
)

print(comparison)

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "svm_knn_comparison.csv"),
    index=False,
)


# ---------------------------------------------------------
# 12. Comparison Visualization
# ---------------------------------------------------------

comparison.set_index("Model").plot(kind="bar")

plt.title("SVM vs KNN Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "svm_knn_comparison.png"),
    dpi=150,
)

plt.close()

print("\nW3D4 execution completed successfully.")
print("Evidence saved in:", OUTPUT_DIR)