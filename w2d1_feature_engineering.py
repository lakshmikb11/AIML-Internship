"""W2D1: Feature engineering, encoding, scaling and feature selection."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

DATA_PATH = Path("data/indian population.csv")
OUTPUT_DIR = Path("output_evidence/w2d1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_features(df: pd.DataFrame) -> None:
    """Apply three categorical encoding techniques and save evidence."""
    data = df[["Level", "TRU"]].dropna()

    label = LabelEncoder().fit_transform(data["Level"])
    one_hot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_result = one_hot.fit_transform(data[["TRU"]])
    ordinal = OrdinalEncoder().fit_transform(data[["Level"]])

    print("\nEncoding results:")
    print("LabelEncoder:", label[:5])
    print("OneHotEncoder shape:", one_hot_result.shape)
    print("OrdinalEncoder:", ordinal[:5])

    pd.DataFrame(
        {
            "LabelEncoder": label,
            "OrdinalEncoder": ordinal.ravel(),
        }
    ).to_csv(OUTPUT_DIR / "encoding_results.csv", index=False)

    print("\nTrade-offs:")
    print("- LabelEncoder: simple, but may imply false ordering.")
    print("- OneHotEncoder: avoids ordering, but increases feature count.")
    print("- OrdinalEncoder: useful when categories have a meaningful order.")


def plot_scalers(series: pd.Series) -> None:
    """Compare Standard, Min-Max and Robust scaling."""
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, (name, scaler) in zip(axes, scalers.items()):
        scaled = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).ravel()
        sns.histplot(scaled, bins=20, kde=True, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Scaled value")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scaling_comparison.png", dpi=300)
    plt.close(fig)


def select_features(df: pd.DataFrame) -> None:
    """Select and save the five strongest features for total population."""
    features = [
        "TOT_M",
        "TOT_F",
        "P_06",
        "P_LIT",
        "TOT_WORK_P",
        "NON_WORK_P",
        "P_SC",
    ]

    data = df[features + ["TOT_P"]].dropna()
    selector = SelectKBest(score_func=f_regression, k=5)
    selector.fit(data[features], data["TOT_P"])

    scores = pd.Series(selector.scores_, index=features).nlargest(5)
    print("\nTop 5 SelectKBest features:")
    print(scores)

    scores.to_csv(OUTPUT_DIR / "top_5_features.csv", header=["score"])


def main() -> None:
    """Run all W2D1 feature engineering tasks."""
    df = pd.read_csv(DATA_PATH)
    print("Dataset Shape:", df.shape)

    encode_features(df)
    plot_scalers(df["TOT_P"].dropna())
    select_features(df)

    print(f"\nW2D1 completed successfully. Evidence saved in: {OUTPUT_DIR}")
    print("Note: Scaling and SelectKBest use the full dataset for this EDA demo.")


if __name__ == "__main__":
    main()