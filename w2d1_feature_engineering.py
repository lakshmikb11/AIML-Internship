
"""W2D1/W2D2: Feature engineering, encoding, scaling and feature selection."""

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

    # LabelEncoder converts categories into integer labels.
    label = LabelEncoder().fit_transform(data["Level"])

    # OneHotEncoder creates separate binary columns for each category.
    one_hot = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )
    one_hot_result = one_hot.fit_transform(data[["TRU"]])

    # OrdinalEncoder converts categories into numerical values.
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
    ).to_csv(
        OUTPUT_DIR / "encoding_results.csv",
        index=False,
    )

    print("\nTrade-offs:")
    print(
        "- LabelEncoder: simple, but may imply false ordering."
    )
    print(
        "- OneHotEncoder: avoids ordering, but increases feature count."
    )
    print(
        "- OrdinalEncoder: useful when categories have a meaningful order."
    )


def plot_scalers(df: pd.DataFrame) -> None:
    """Compare feature distributions before and after scaling."""
    features = [
        "TOT_M",
        "TOT_F",
        "P_06",
        "P_LIT",
        "TOT_WORK_P",
        "NON_WORK_P",
        "P_SC",
    ]

    data = df[features].dropna()

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    # Apply each scaler to the numeric predictor features.
    for name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(data)

        scaled_df = pd.DataFrame(
            scaled_data,
            columns=features,
            index=data.index,
        )

        # Create before/after distribution plots for each feature.
        fig, axes = plt.subplots(
            len(features),
            2,
            figsize=(12, 4 * len(features)),
        )

        for row, feature in enumerate(features):
            # Original distribution before scaling.
            sns.histplot(
                data[feature],
                bins=20,
                kde=True,
                ax=axes[row, 0],
            )
            axes[row, 0].set_title(
                f"{feature} - Before Scaling"
            )
            axes[row, 0].set_xlabel(feature)

            # Distribution after scaling.
            sns.histplot(
                scaled_df[feature],
                bins=20,
                kde=True,
                ax=axes[row, 1],
            )
            axes[row, 1].set_title(
                f"{feature} - {name}"
            )
            axes[row, 1].set_xlabel("Scaled value")

        fig.suptitle(
            f"{name}: Feature Distributions Before and After Scaling",
            fontsize=14,
        )
        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"{name.lower()}_feature_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def select_features(df: pd.DataFrame) -> None:
    """Select and document the five strongest features for total population."""
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

    # f_regression is appropriate because TOT_P is a numeric target.
    selector = SelectKBest(
        score_func=f_regression,
        k=5,
    )
    selector.fit(
        data[features],
        data["TOT_P"],
    )

    # Rank all features by their SelectKBest score.
    scores = pd.Series(
        selector.scores_,
        index=features,
    ).sort_values(ascending=False)

    print("\nTop 5 SelectKBest features:")
    print(scores.head(5))

    # Save the top five feature scores.
    top_5 = scores.head(5)

    top_5.to_csv(
        OUTPUT_DIR / "top_5_features.csv",
        header=["score"],
    )

    # Explain why each selected feature matters.
    feature_explanations = {
        "TOT_M": (
            "Total male population is a major component "
            "of total population."
        ),
        "TOT_F": (
            "Total female population is a major component "
            "of total population."
        ),
        "P_06": (
            "Population aged 0-6 represents an important "
            "demographic group."
        ),
        "P_LIT": (
            "Literate population provides information about "
            "the size of the literate population."
        ),
        "TOT_WORK_P": (
            "Total working population represents the "
            "economically active population."
        ),
        "NON_WORK_P": (
            "Non-working population provides information "
            "about the population not engaged in work."
        ),
        "P_SC": (
            "Scheduled Caste population represents an "
            "important demographic group."
        ),
    }

    explanation_rows = []

    for feature, score in top_5.items():
        explanation_rows.append(
            {
                "Feature": feature,
                "SelectKBest_Score": score,
                "Why_it_matters": feature_explanations.get(
                    feature,
                    (
                        "This feature has a strong statistical "
                        "relationship with total population."
                    ),
                ),
            }
        )

    pd.DataFrame(explanation_rows).to_csv(
        OUTPUT_DIR / "top_5_feature_explanations.csv",
        index=False,
    )


def main() -> None:
    """Run all W2D2 feature engineering tasks."""
    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)

    encode_features(df)

    # Scale numeric predictor features, not the target.
    plot_scalers(df)

    select_features(df)

    print(
        f"\nW2D2 completed successfully. "
        f"Evidence saved in: {OUTPUT_DIR}"
    )

    print(
        "Note: Scaling and SelectKBest use the full dataset "
        "for this EDA demonstration."
    )


if __name__ == "__main__":
    main()

