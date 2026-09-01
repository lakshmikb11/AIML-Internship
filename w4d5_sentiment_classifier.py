"""
W4D5: Sentiment Classifier — Logistic Regression vs Random Forest

This script:
1. Creates a reproducible binary sentiment dataset.
2. Converts text into TF-IDF features.
3. Trains Logistic Regression.
4. Prints a classification report.
5. Generates a confusion matrix and ROC-AUC curve.
6. Trains Random Forest.
7. Compares accuracy, precision, recall, and ROC-AUC.
8. Saves evaluation evidence to output_evidence/w4d5/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42

OUTPUT_DIR = Path("output_evidence/w4d5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Create binary sentiment dataset
# ---------------------------------------------------------
# The dataset contains clearly labeled positive and negative
# reviews with varied wording to provide a reproducible
# binary sentiment classification task.

positive_templates = [
    "I loved this product because it was excellent",
    "This product was amazing and worked perfectly",
    "I had a wonderful experience with this service",
    "The quality was excellent and I am very satisfied",
    "This was a fantastic experience and I recommend it",
    "I really enjoyed using this product",
    "The service was great and very helpful",
    "I am happy with this purchase",
    "The movie was brilliant and enjoyable",
    "This product exceeded my expectations",
    "The experience was pleasant and satisfying",
    "I would definitely recommend this product",
    "Everything worked perfectly and I am satisfied",
    "This was one of the best experiences",
    "The product is reliable and excellent",
    "I am very pleased with the quality",
    "The service was outstanding and friendly",
    "This purchase was worth the money",
    "I enjoyed the excellent customer service",
    "The product quality is fantastic",
]

negative_templates = [
    "I hated this product because it was terrible",
    "This product was awful and did not work",
    "I had a horrible experience with this service",
    "The quality was poor and I am very dissatisfied",
    "This was a terrible experience and I do not recommend it",
    "I really disliked using this product",
    "The service was bad and very unhelpful",
    "I am unhappy with this purchase",
    "The movie was boring and disappointing",
    "This product failed to meet my expectations",
    "The experience was unpleasant and frustrating",
    "I would not recommend this product",
    "Nothing worked properly and I am dissatisfied",
    "This was one of the worst experiences",
    "The product is unreliable and terrible",
    "I am very disappointed with the quality",
    "The service was horrible and unfriendly",
    "This purchase was a waste of money",
    "I disliked the poor customer service",
    "The product quality is awful",
]

positive_prefixes = [
    "Overall,",
    "Honestly,",
    "In my opinion,",
    "From my experience,",
    "For me,",
]

negative_prefixes = [
    "Overall,",
    "Honestly,",
    "In my opinion,",
    "From my experience,",
    "For me,",
]

positive_suffixes = [
    "I would buy it again.",
    "It made me very happy.",
    "I am completely satisfied.",
    "I would recommend it to others.",
    "It was a great choice.",
]

negative_suffixes = [
    "I would never buy it again.",
    "It made me very disappointed.",
    "I am completely dissatisfied.",
    "I would warn others about it.",
    "It was a very bad choice.",
]


positive_texts = []

for text in positive_templates:
    for prefix in positive_prefixes:
        for suffix in positive_suffixes:
            positive_texts.append(
                f"{prefix} {text.lower()}. {suffix}"
            )


negative_texts = []

for text in negative_templates:
    for prefix in negative_prefixes:
        for suffix in negative_suffixes:
            negative_texts.append(
                f"{prefix} {text.lower()}. {suffix}"
            )


data = pd.DataFrame(
    {
        "text": positive_texts + negative_texts,
        "sentiment": (
            [1] * len(positive_texts)
            + [0] * len(negative_texts)
        ),
    }
)

print(f"Dataset shape: {data.shape}")
print(
    f"Positive samples: "
    f"{(data['sentiment'] == 1).sum()}"
)
print(
    f"Negative samples: "
    f"{(data['sentiment'] == 0).sum()}"
)


# ---------------------------------------------------------
# 3. Save dataset evidence
# ---------------------------------------------------------

data.to_csv(
    OUTPUT_DIR / "sentiment_dataset.csv",
    index=False,
)


# ---------------------------------------------------------
# 4. Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["sentiment"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=data["sentiment"],
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ---------------------------------------------------------
# 5. TF-IDF feature extraction
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(
    f"TF-IDF training shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"TF-IDF testing shape: "
    f"{X_test_tfidf.shape}"
)


# ---------------------------------------------------------
# 6. Logistic Regression
# ---------------------------------------------------------

logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000,
)

logistic_model.fit(
    X_train_tfidf,
    y_train,
)

logistic_predictions = logistic_model.predict(
    X_test_tfidf
)

logistic_probabilities = (
    logistic_model.predict_proba(X_test_tfidf)[:, 1]
)


print("\nLogistic Regression Classification Report")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["Negative", "Positive"],
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 7. Logistic Regression metrics
# ---------------------------------------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions,
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0,
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0,
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities,
)


print(
    f"Logistic Accuracy:  "
    f"{logistic_accuracy:.4f}"
)

print(
    f"Logistic Precision: "
    f"{logistic_precision:.4f}"
)

print(
    f"Logistic Recall:    "
    f"{logistic_recall:.4f}"
)

print(
    f"Logistic ROC-AUC:   "
    f"{logistic_roc_auc:.4f}"
)


# ---------------------------------------------------------
# 8. Logistic Regression confusion matrix
# ---------------------------------------------------------

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions,
)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=logistic_cm,
    display_labels=["Negative", "Positive"],
).plot(ax=ax)

ax.set_title(
    "Logistic Regression - Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "logistic_confusion_matrix.png",
    dpi=150,
)

plt.close(fig)


# ---------------------------------------------------------
# 9. Logistic Regression ROC-AUC curve
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

RocCurveDisplay.from_predictions(
    y_test,
    logistic_probabilities,
    name="Logistic Regression",
    ax=ax,
)

ax.set_title(
    "Logistic Regression - ROC-AUC Curve"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "logistic_roc_auc_curve.png",
    dpi=150,
)

plt.close(fig)


# ---------------------------------------------------------
# 10. Random Forest
# ---------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

random_forest.fit(
    X_train_tfidf,
    y_train,
)

rf_predictions = random_forest.predict(
    X_test_tfidf
)

rf_probabilities = (
    random_forest.predict_proba(X_test_tfidf)[:, 1]
)


print("\nRandom Forest Classification Report")
print("-----------------------------------")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Negative", "Positive"],
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 11. Random Forest metrics
# ---------------------------------------------------------

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions,
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    zero_division=0,
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    zero_division=0,
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities,
)


print(
    f"Random Forest Accuracy:  "
    f"{rf_accuracy:.4f}"
)

print(
    f"Random Forest Precision: "
    f"{rf_precision:.4f}"
)

print(
    f"Random Forest Recall:    "
    f"{rf_recall:.4f}"
)

print(
    f"Random Forest ROC-AUC:   "
    f"{rf_roc_auc:.4f}"
)


# ---------------------------------------------------------
# 12. Random Forest confusion matrix
# ---------------------------------------------------------

rf_cm = confusion_matrix(
    y_test,
    rf_predictions,
)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Negative", "Positive"],
).plot(ax=ax)

ax.set_title(
    "Random Forest - Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "random_forest_confusion_matrix.png",
    dpi=150,
)

plt.close(fig)


# ---------------------------------------------------------
# 13. Combined ROC-AUC comparison
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

RocCurveDisplay.from_predictions(
    y_test,
    logistic_probabilities,
    name="Logistic Regression",
    ax=ax,
)

RocCurveDisplay.from_predictions(
    y_test,
    rf_probabilities,
    name="Random Forest",
    ax=ax,
)

ax.set_title(
    "ROC-AUC Comparison: Logistic Regression vs Random Forest"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "roc_auc_comparison.png",
    dpi=150,
)

plt.close(fig)


# ---------------------------------------------------------
# 14. Model comparison
# ---------------------------------------------------------

comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
        ],
        "Accuracy": [
            logistic_accuracy,
            rf_accuracy,
        ],
        "Precision": [
            logistic_precision,
            rf_precision,
        ],
        "Recall": [
            logistic_recall,
            rf_recall,
        ],
        "ROC_AUC": [
            logistic_roc_auc,
            rf_roc_auc,
        ],
    }
)


print("\nModel Comparison")
print("----------------")

print(
    comparison.to_string(index=False)
)


comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# 15. Classification metrics evidence
# ---------------------------------------------------------

metrics_evidence = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "ROC_AUC",
        ],
        "Logistic_Regression": [
            logistic_accuracy,
            logistic_precision,
            logistic_recall,
            logistic_roc_auc,
        ],
        "Random_Forest": [
            rf_accuracy,
            rf_precision,
            rf_recall,
            rf_roc_auc,
        ],
    }
)


metrics_evidence.to_csv(
    OUTPUT_DIR / "classification_metrics.csv",
    index=False,
)


# ---------------------------------------------------------
# 16. Prediction evidence
# ---------------------------------------------------------

prediction_evidence = pd.DataFrame(
    {
        "text": X_test.values,
        "actual_sentiment": y_test.values,
        "logistic_prediction": logistic_predictions,
        "random_forest_prediction": rf_predictions,
        "logistic_probability_positive": (
            logistic_probabilities
        ),
        "random_forest_probability_positive": (
            rf_probabilities
        ),
    }
)


prediction_evidence.to_csv(
    OUTPUT_DIR / "prediction_evidence.csv",
    index=False,
)


# ---------------------------------------------------------
# 17. Final verification
# ---------------------------------------------------------

required_files = [
    "sentiment_dataset.csv",
    "classification_metrics.csv",
    "model_comparison.csv",
    "prediction_evidence.csv",
    "logistic_confusion_matrix.png",
    "random_forest_confusion_matrix.png",
    "logistic_roc_auc_curve.png",
    "roc_auc_comparison.png",
]

all_files_exist = all(
    (OUTPUT_DIR / filename).exists()
    for filename in required_files
)


print(
    f"\nAll evidence files created: "
    f"{all_files_exist}"
)

print(f"Evidence saved to: {OUTPUT_DIR}")
print("W4D5 execution completed successfully.")