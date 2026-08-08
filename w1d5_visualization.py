"""W1D5: Data Visualisation using Matplotlib and Seaborn."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = "data/indian population.csv"
OUTPUT_DIR = "output_evidence/w1d5_plots"
REQUIRED_COLUMNS = {"District", "TOT_P", "TOT_M", "TOT_F"}


def save_plot(filename):
    """Save and close the current plot."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def main():
    """Load data and generate EDA visualisations."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["TOT_P", "TOT_M", "TOT_F"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())

    # 1. Top 10 districts by population.
    top_districts = df.nlargest(10, "TOT_P")
    sns.barplot(data=top_districts, x="TOT_P", y="District")
    plt.title("Top 10 Districts by Total Population")
    plt.xlabel("Total Population")
    plt.ylabel("District")
    save_plot("top_10_district_population.png")

    # 2. Distribution of total population.
    sns.histplot(data=df, x="TOT_P", bins=20, kde=True)
    plt.title("Distribution of Total Population")
    plt.xlabel("Total Population")
    plt.ylabel("Number of Districts")
    save_plot("population_distribution.png")

    # 3. Relationship between male and female population.
    sns.scatterplot(data=df, x="TOT_M", y="TOT_F")
    plt.title("Male Population vs Female Population")
    plt.xlabel("Male Population")
    plt.ylabel("Female Population")
    save_plot("male_vs_female_population.png")

    # 4. Correlation between population variables.
    corr = df[["TOT_P", "TOT_M", "TOT_F"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Population Correlation")
    save_plot("population_correlation_heatmap.png")

    print("\nW1D5 visualisation completed successfully.")
    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()