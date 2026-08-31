
"""
W3D5: Hyperparameter Tuning — GridSearch & RandomSearch

Cynaris AI/ML Internship

This script compares GridSearchCV and RandomizedSearchCV
for tuning SVM and KNN classifiers using the Iris dataset.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
OUTPUT_DIR = "output_evidence/w3d5"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# SVM — GridSearchCV
# =========================================================

print("\n========== SVM — GRID SEARCH ==========")


svm_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ]
)


svm_grid_params = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", "auto"],
}


svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_grid_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)


svm_grid.fit(X_train, y_train)

best_svm_grid = svm_grid.best_estimator_

print("Best parameters:", svm_grid.best_params_)
print("Best CV accuracy:", svm_grid.best_score_)


# =========================================================
# SVM — RandomizedSearchCV
# =========================================================

print("\n========== SVM — RANDOM SEARCH ==========")


svm_random_params = {
    "svm__C": [0.01, 0.1, 1, 10, 100, 1000],
    "svm__kernel": ["linear", "rbf", "poly", "sigmoid"],
    "svm__gamma": ["scale", "auto"],
}


svm_random = RandomizedSearchCV(
    estimator=svm_pipeline,
    param_distributions=svm_random_params,
    n_iter=15,
    cv=5,
    scoring="accuracy",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


svm_random.fit(X_train, y_train)

best_svm_random = svm_random.best_estimator_

print("Best parameters:", svm_random.best_params_)
print("Best CV accuracy:", svm_random.best_score_)


# =========================================================
# KNN — GridSearchCV
# =========================================================

print("\n========== KNN — GRID SEARCH ==========")


knn_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)


knn_grid_params = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "minkowski"],
}


knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_grid_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)


knn_grid.fit(X_train, y_train)

best_knn_grid = knn_grid.best_estimator_

print("Best parameters:", knn_grid.best_params_)
print("Best CV accuracy:", knn_grid.best_score_)


# =========================================================
# KNN — RandomizedSearchCV
# =========================================================

print("\n========== KNN — RANDOM SEARCH ==========")


knn_random_params = {
    "knn__n_neighbors": list(range(1, 21)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "minkowski"],
}


knn_random = RandomizedSearchCV(
    estimator=knn_pipeline,
    param_distributions=knn_random_params,
    n_iter=15,
    cv=5,
    scoring="accuracy",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


knn_random.fit(X_train, y_train)

best_knn_random = knn_random.best_estimator_

print("Best parameters:", knn_random.best_params_)
print("Best CV accuracy:", knn_random.best_score_)


# =========================================================
# Model Evaluation Helper
# =========================================================


def evaluate_model(model_name, model):
    """Evaluate a tuned model on the unseen test set."""

    predictions = model.predict(X_test)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "F1-Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# 3. Evaluate Tuned Models
# ---------------------------------------------------------

results = [
    evaluate_model("SVM - GridSearch", best_svm_grid),
    evaluate_model("SVM - RandomSearch", best_svm_random),
    evaluate_model("KNN - GridSearch", best_knn_grid),
    evaluate_model("KNN - RandomSearch", best_knn_random),
]


results_df = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# ---------------------------------------------------------
# 4. Save Results
# ---------------------------------------------------------

results_path = os.path.join(
    OUTPUT_DIR,
    "hyperparameter_tuning_results.csv",
)

results_df.to_csv(results_path, index=False)


# ---------------------------------------------------------
# 5. Save Best Parameters
# ---------------------------------------------------------

best_parameters = pd.DataFrame(
    [
        {
            "Model": "SVM - GridSearch",
            "Best Parameters": str(svm_grid.best_params_),
            "Best CV Accuracy": svm_grid.best_score_,
        },
        {
            "Model": "SVM - RandomSearch",
            "Best Parameters": str(svm_random.best_params_),
            "Best CV Accuracy": svm_random.best_score_,
        },
        {
            "Model": "KNN - GridSearch",
            "Best Parameters": str(knn_grid.best_params_),
            "Best CV Accuracy": knn_grid.best_score_,
        },
        {
            "Model": "KNN - RandomSearch",
            "Best Parameters": str(knn_random.best_params_),
            "Best CV Accuracy": knn_random.best_score_,
        },
    ]
)

best_parameters_path = os.path.join(
    OUTPUT_DIR,
    "best_hyperparameters.csv",
)

best_parameters.to_csv(
    best_parameters_path,
    index=False,
)


# ---------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar")

plt.title("GridSearch vs RandomSearch Performance")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "hyperparameter_tuning_comparison.png",
)

plt.savefig(plot_path, dpi=150)
plt.close()


print("\nW3D5 execution completed successfully.")
print("Evidence saved in:", OUTPUT_DIR)

