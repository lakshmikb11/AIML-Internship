import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned Indian population dataset
df = pd.read_csv("india_population_cleaned.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# TASK 1: BASIC EDA
# ============================================================

# 1. Statistical summary
print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

# 2. Dataset information
print("\n===== DATASET INFORMATION =====")
df.info()

# 3. Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# ============================================================
# TASK 2: DISTRIBUTION OF NUMERIC COLUMNS
# ============================================================

# Identifier/code columns are excluded because
# they are not meaningful measurements.
exclude_columns = [
    "State",
    "District",
    "Subdistt",
    "Town/Village",
    "Ward",
    "EB"
]

# Select numeric columns
numeric_columns = df.select_dtypes(include="number").columns

# Remove identifier/code columns
numeric_columns = [
    col for col in numeric_columns
    if col not in exclude_columns
]

# Plot distributions of numeric columns
df[numeric_columns].hist(
    figsize=(20, 25),
    bins=20
)

plt.suptitle(
    "Distribution of Numeric Variables",
    fontsize=16,
    y=1.02
)

plt.tight_layout()

# Save the distribution plot
plt.savefig(
    "output_evidence/w1d4_numeric_distributions.png",
    dpi=150
)

print("\nDistribution plot saved to:")
print("output_evidence/w1d4_numeric_distributions.png")

# ============================================================
# TASK 2: CORRELATION HEATMAP
# ============================================================

# Select important population and workforce columns
correlation_columns = [
    "TOT_P",
    "TOT_M",
    "TOT_F",
    "P_06",
    "P_SC",
    "P_ST",
    "P_LIT",
    "P_ILL",
    "TOT_WORK_P",
    "TOT_WORK_M",
    "TOT_WORK_F",
    "MAINWORK_P",
    "MARGWORK_P",
    "NON_WORK_P"
]

# Calculate correlation matrix
correlation_matrix = df[correlation_columns].corr()

# Create heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Population and Workforce Variables")
plt.tight_layout()

# Save the heatmap
plt.savefig(
    "output_evidence/w1d4_correlation_heatmap.png",
    dpi=150
)

print("\nCorrelation heatmap saved to:")
print("output_evidence/w1d4_correlation_heatmap.png")

plt.close()

# ============================================================
# TASK 2: TOP-10 CATEGORY COUNTS
# ============================================================

# Count the categories in the TRU column
top_categories = df["TRU"].value_counts().head(10)

# Plot top-10 categories
plt.figure(figsize=(10, 6))

top_categories.plot(kind="bar")

plt.title("Top-10 Category Counts - TRU")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()

# Save the category count plot
plt.savefig(
    "output_evidence/w1d4_top10_category_counts.png",
    dpi=150
)

print("\nTop-10 category count plot saved to:")
print("output_evidence/w1d4_top10_category_counts.png")

plt.close()
