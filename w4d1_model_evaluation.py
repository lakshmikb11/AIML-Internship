"""
W4D1: Model Evaluation Metrics

Topics covered:
- Train/Test Split
- Precision
- Recall
- ROC-AUC
- ROC Curve
- Evaluation Evidence
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


EVIDENCE_DIR = Path("output_evidence/w4d1")


def load_dataset():
    """Load and return the Breast Cancer Wisconsin dataset."""

    data = load_breast_cancer()

    X = data.data
    y = data.target

    print("Dataset shape:", X.shape)
    print("Target classes:", data.target_names)

    return X, y


def train_model(X, y):
    """Split the dataset and train a Logistic Regression model."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using Precision, Recall, and ROC-AUC."""

    # Generate predicted class labels.
    y_pred = model.predict(X_test)

    # Generate probability scores for the positive class.
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics.
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nTest Set Evaluation")
    print("-------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    return precision, recall, roc_auc, y_test, y_prob


def save_evidence(precision, recall, roc_auc, y_test, y_prob):
    """Save evaluation metrics and ROC curve as evidence."""

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Save numerical metrics.
    metrics_file = EVIDENCE_DIR / "evaluation_metrics.txt"

    with open(metrics_file, "w", encoding="utf-8") as file:
        file.write("W4D1 Model Evaluation Results\n")
        file.write("=============================\n\n")
        file.write("Dataset: Breast Cancer Wisconsin\n")
        file.write("Model: Logistic Regression\n\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall:    {recall:.4f}\n")
        file.write(f"ROC-AUC:   {roc_auc:.4f}\n")

    # Generate ROC curve.
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("W4D1 ROC Curve")
    plt.legend()
    plt.grid(True)

    roc_file = EVIDENCE_DIR / "roc_curve.png"
    plt.savefig(roc_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nEvidence saved:")
    print(f"- {metrics_file}")
    print(f"- {roc_file}")


def main():
    """Run the complete W4D1 evaluation workflow."""

    # Step 1: Load dataset.
    X, y = load_dataset()

    # Step 2: Train the Logistic Regression model.
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # Step 3: Evaluate the trained model.
    precision, recall, roc_auc, y_test, y_prob = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Step 4: Save evaluation evidence.
    save_evidence(
        precision,
        recall,
        roc_auc,
        y_test,
        y_prob
    )


if __name__ == "__main__":
    main()