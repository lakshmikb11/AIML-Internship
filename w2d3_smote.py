"""W2D3: Handling imbalanced data using SMOTE."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("data/indian population.csv")
OUTPUT_DIR = Path("output_evidence/w2d3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and prepare numeric features and target."""

    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)

    label_encoder = LabelEncoder()
    y = pd.Series(
        label_encoder.fit_transform(df["Level"]),
        name="Level",
    )

    X = df.select_dtypes(include="number").copy()

    # State and District are identifiers, not predictive features.
    X = X.drop(columns=["State", "District"], errors="ignore")

    return X, y


def show_distribution(
    y: pd.Series,
    title: str,
    filename: str,
) -> None:
    """Display, save, and print class distribution."""

    distribution = y.value_counts().sort_index()

    print(f"\n{title}:")
    print(distribution)

    ax = distribution.plot(
        kind="bar",
        figsize=(6, 4),
        title=title,
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE using a valid k_neighbors value."""

    minority_count = y_train.value_counts().min()

    if minority_count < 2:
        raise ValueError(
            "SMOTE requires at least 2 samples in the minority class."
        )

    # Only 2 minority samples are available in the training set.
    # Therefore, k_neighbors=1 is required for SMOTE to run.
    # With so few minority samples, synthetic samples may have
    # limited diversity and may not generalize well.
    k_neighbors = min(5, minority_count - 1)

    print(f"\nUsing SMOTE with k_neighbors={k_neighbors}")

    smote = SMOTE(
        random_state=RANDOM_SEED,
        k_neighbors=k_neighbors,
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train,
    )

    return (
        pd.DataFrame(X_resampled, columns=X_train.columns),
        pd.Series(y_resampled, name="Level"),
    )


def save_resampled_data(
    X_resampled: pd.DataFrame,
    y_resampled: pd.Series,
) -> None:
    """Save the SMOTE-resampled training data."""

    resampled = X_resampled.copy()
    resampled["Level"] = y_resampled

    resampled.to_csv(
        OUTPUT_DIR / "smote_resampled_data.csv",
        index=False,
    )


def main() -> None:
    """Run the complete W2D3 SMOTE workflow."""

    X, y = load_and_prepare_data()

    show_distribution(
        y,
        "Original class distribution",
        "original_class_distribution.png",
    )

    # Stratified split keeps both classes represented in the split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("\nTraining class distribution before SMOTE:")
    print(y_train.value_counts().sort_index())

    # Apply SMOTE ONLY to training data to prevent data leakage.
    X_resampled, y_resampled = apply_smote(
        X_train,
        y_train,
    )

    print("\nTraining class distribution after SMOTE:")
    print(y_resampled.value_counts().sort_index())

    show_distribution(
        y_resampled,
        "SMOTE class distribution",
        "smote_class_distribution.png",
    )

    save_resampled_data(
        X_resampled,
        y_resampled,
    )

    print(
        f"\nW2D3 completed successfully. "
        f"Evidence saved in: {OUTPUT_DIR}"
    )

    print(
        "SMOTE was applied only to the training data "
        "to avoid data leakage."
    )


if __name__ == "__main__":
    main()