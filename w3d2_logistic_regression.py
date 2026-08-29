"""
W3D2: Logistic Regression & Classification

Train and evaluate a Logistic Regression classifier
using the Iris dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("output_evidence/w3d2")


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the Iris classification dataset
    iris = load_iris()

    X = pd.DataFrame(
        iris.data,
        columns=iris.feature_names,
    )
    y = pd.Series(iris.target, name="target")

    print("Dataset shape:", X.shape)
    print("Features:", list(X.columns))
    print("Classes:", list(iris.target_names))

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Scale numerical features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train)

    # Print model coefficients and intercept
    print("\nModel coefficients:")
    print(model.coef_)

    print("\nModel intercept:")
    print(model.intercept_)

    # Generate predictions
    y_pred = model.predict(X_test_scaled)
    y_probability = model.predict_proba(X_test_scaled)

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\nClassification Metrics")
    print("----------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Classification report
    print("\nClassification Report")
    print("---------------------")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names,
            zero_division=0,
        )
    )

    # Save metrics
    metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
            ],
            "Score": [
                accuracy,
                precision,
                recall,
                f1,
            ],
        }
    )

    metrics.to_csv(
        OUTPUT_DIR / "classification_metrics.csv",
        index=False,
    )

    # Save prediction probabilities
    probabilities = pd.DataFrame(
        y_probability,
        columns=[
            f"Probability_{name}"
            for name in iris.target_names
        ],
    )

    probabilities["Actual"] = y_test.to_numpy()
    probabilities["Predicted"] = y_pred

    probabilities.to_csv(
        OUTPUT_DIR / "prediction_probabilities.csv",
        index=False,
    )

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("W3D2 Logistic Regression - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    # Create decision boundary using first two features
    X_boundary = X.iloc[:, :2]

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_boundary,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    boundary_scaler = StandardScaler()

    Xb_train_scaled = boundary_scaler.fit_transform(Xb_train)

    boundary_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    boundary_model.fit(
        Xb_train_scaled,
        yb_train,
    )

    # Create mesh grid
    x_min = Xb_train_scaled[:, 0].min() - 1
    x_max = Xb_train_scaled[:, 0].max() + 1
    y_min = Xb_train_scaled[:, 1].min() - 1
    y_max = Xb_train_scaled[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[
        xx.ravel(),
        yy.ravel(),
    ]

    predictions = boundary_model.predict(grid)

    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.25,
    )

    for class_value, class_name in enumerate(
        iris.target_names
    ):
        mask = yb_train.to_numpy() == class_value

        plt.scatter(
            Xb_train_scaled[mask, 0],
            Xb_train_scaled[mask, 1],
            label=class_name,
        )

    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(
        "W3D2 Logistic Regression - Decision Boundaries"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "decision_boundaries.png",
        dpi=300,
    )

    plt.close()

    print("\nOutput evidence saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()