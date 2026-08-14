
"""W2D5: Titanic end-to-end preprocessing pipeline."""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DATA_PATH = Path("data/titanic.csv")
OUTPUT_DIR = Path("output_evidence/w2d5")
OUTPUT_PATH = OUTPUT_DIR / "titanic_ml_ready.csv"
PIPELINE_PATH = OUTPUT_DIR / "titanic_preprocessor.joblib"


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)

    # Create output folder before saving evidence
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Lightweight EDA
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNumeric summary:")
    print(df[["Age", "Fare", "SibSp", "Parch"]].describe())

    print("\nSex distribution:")
    print(df["Sex"].value_counts())

    # Save EDA plot
    df[["Age", "Fare"]].hist(figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_distributions.png")
    plt.close()

    # 3. Separate target and remove unused columns
    # PassengerId, Name, Ticket and Cabin are excluded because they are
    # identifiers/high-cardinality fields or contain substantial missing data.
    X = df.drop(
        columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"]
    )
    y = df["Survived"]

    # Pclass is categorical/ordinal, so it is not scaled as continuous data.
    numeric = ["Age", "SibSp", "Parch", "Fare"]
    categorical = ["Pclass", "Sex", "Embarked"]

    # 4. Split before fitting preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # 5. Preprocessing pipeline
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

    # Fit only on training data
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    # 6. Transform complete dataset using training-fitted pipeline
    X_ready = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()
    ready_df = pd.DataFrame(X_ready, columns=feature_names)
    ready_df["Survived"] = y.values

    # 7. Export ML-ready data and fitted pipeline
    ready_df.to_csv(OUTPUT_PATH, index=False)
    joblib.dump(preprocessor, PIPELINE_PATH)

    # 8. Validate output
    print("\nML-ready shape:", ready_df.shape)
    print("Missing values:", ready_df.isnull().sum().sum())
    print("Saved:", OUTPUT_PATH)
    print("Pipeline saved:", PIPELINE_PATH)
    print("\nW2D5 completed successfully.")


if __name__ == "__main__":
    main()
