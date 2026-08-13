"""
W2D4: Train/Test Split & Cross-Validation

This script demonstrates:
1. Train/test splitting
2. Stratified splitting using binned continuous target values
3. K-Fold cross-validation
4. Model evaluation using cross-validation
5. Prevention of data leakage using a pipeline
6. Saving output evidence

Dataset:
    data/indian population.csv

Target:
    TOT_P
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Reproducibility
RANDOM_SEED = 42

# File paths
DATA_PATH = Path("data/indian population.csv")
OUTPUT_DIR = Path("output_evidence/w2d4")

# Target column
TARGET_COLUMN = "TOT_P"

# Number of cross-validation folds
CV_FOLDS = 5


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the population dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")

    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare numeric predictor features.

    TOT_P is excluded from X because it is the target.
    Non-numeric columns are excluded from this numerical demonstration.
    """
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    feature_columns = [
        column for column in numeric_columns
        if column != TARGET_COLUMN
    ]

    if not feature_columns:
        raise ValueError("No numeric predictor features are available.")

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    if X.isna().any().any():
        raise ValueError("Predictor features contain missing values.")

    return X, y, feature_columns


def create_target_bins(y: pd.Series) -> pd.Series:
    """
    Create quantile bins from the continuous target.

    The bins are used only for stratified train/test splitting.
    The original continuous target remains unchanged for model training.
    """
    try:
        bins = pd.qcut(
            y,
            q=5,
            labels=False,
            duplicates="drop",
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create target bins for stratification."
        ) from exc

    if bins.nunique() < 2:
        raise ValueError(
            "Unable to create enough target bins for stratification."
        )

    return bins


def perform_train_test_split(X, y):
    """
    Split the data into training and testing sets.

    Because TOT_P is continuous, quantile bins are used only to
    stratify the split and maintain a representative target distribution.
    """
    target_bins = create_target_bins(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=target_bins,
    )

    return X_train, X_test, y_train, y_test


def build_pipeline():
    """
    Build a leakage-safe machine learning pipeline.

    StandardScaler is fitted separately inside each training fold,
    preventing information from the validation/test data from
    influencing the scaling process.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def perform_cross_validation(X_train, y_train):
    """Perform 5-fold cross-validation using training data only."""
    pipeline = build_pipeline()

    kfold = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    r2_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring="r2",
    )

    mae_scores = -cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring="neg_mean_absolute_error",
    )

    rmse_scores = -cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring="neg_root_mean_squared_error",
    )

    return r2_scores, mae_scores, rmse_scores


def evaluate_test_set(X_train, X_test, y_train, y_test):
    """Train the final pipeline and evaluate it on the unseen test set."""
    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


def save_split_distribution(y_train, y_test):
    """Save a plot comparing training and testing target distributions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(y_train, bins=20, alpha=0.7, label="Training")
    plt.hist(y_test, bins=20, alpha=0.7, label="Testing")

    plt.xlabel("TOT_P")
    plt.ylabel("Frequency")
    plt.title("Train/Test Target Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "train_test_target_distribution.png",
        dpi=150,
    )
    plt.close()


def save_cross_validation_results(
    r2_scores,
    mae_scores,
    rmse_scores,
):
    """Save cross-validation fold results as CSV."""
    results = pd.DataFrame(
        {
            "fold": range(1, CV_FOLDS + 1),
            "r2": r2_scores,
            "mae": mae_scores,
            "rmse": rmse_scores,
        }
    )

    results.to_csv(
        OUTPUT_DIR / "cross_validation_results.csv",
        index=False,
    )


def save_summary(
    X,
    X_train,
    X_test,
    r2_scores,
    mae_scores,
    rmse_scores,
    test_mae,
    test_rmse,
    test_r2,
):
    """Save train/test split and cross-validation summary."""
    summary = pd.DataFrame(
        {
            "metric": [
                "total_samples",
                "training_samples",
                "testing_samples",
                "feature_count",
                "cv_folds",
                "cv_mean_r2",
                "cv_std_r2",
                "cv_mean_mae",
                "cv_mean_rmse",
                "test_mae",
                "test_rmse",
                "test_r2",
            ],
            "value": [
                len(X),
                len(X_train),
                len(X_test),
                X.shape[1],
                CV_FOLDS,
                r2_scores.mean(),
                r2_scores.std(),
                mae_scores.mean(),
                rmse_scores.mean(),
                test_mae,
                test_rmse,
                test_r2,
            ],
        }
    )

    summary.to_csv(
        OUTPUT_DIR / "train_test_cv_summary.csv",
        index=False,
    )


def main():
    """Run the complete W2D4 workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_dataset(DATA_PATH)

    print(f"Dataset Shape: {df.shape}")

    # Prepare numeric predictors and target
    X, y, feature_columns = prepare_features(df)

    print(f"Number of predictor features: {len(feature_columns)}")
    print(f"Target column: {TARGET_COLUMN}")

    # Split before cross-validation
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X,
        y,
    )

    print("\nTrain/Test Split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Cross-validation is performed only on training data
    r2_scores, mae_scores, rmse_scores = perform_cross_validation(
        X_train,
        y_train,
    )

    print("\n5-Fold Cross-Validation:")
    print("R² scores:", r2_scores)
    print("Mean R²:", r2_scores.mean())
    print("R² standard deviation:", r2_scores.std())

    print("MAE scores:", mae_scores)
    print("Mean MAE:", mae_scores.mean())

    print("RMSE scores:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())

    # Final evaluation on the unseen test set
    test_mae, test_rmse, test_r2 = evaluate_test_set(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nTest Set Evaluation:")
    print("MAE:", test_mae)
    print("RMSE:", test_rmse)
    print("R²:", test_r2)

    # Save output evidence
    save_split_distribution(y_train, y_test)

    save_cross_validation_results(
        r2_scores,
        mae_scores,
        rmse_scores,
    )

    save_summary(
        X,
        X_train,
        X_test,
        r2_scores,
        mae_scores,
        rmse_scores,
        test_mae,
        test_rmse,
        test_r2,
    )

    print("\nW2D4 completed successfully.")
    print(f"Evidence saved in: {OUTPUT_DIR}")
    print("Scaling is performed inside the pipeline to prevent data leakage.")
    print("Cross-validation is performed only on the training data.")
    print(
        "Note: Very high R² may be caused by population-component "
        "features that are mathematically related to TOT_P."
    )


if __name__ == "__main__":
    main()