"""
W4D2: Bias-Variance Tradeoff & Regularisation

This script demonstrates:
1. Linear Regression as a baseline model.
2. Ridge and Lasso regularisation.
3. Systematic hyperparameter tuning using GridSearchCV.
4. Randomized hyperparameter tuning using RandomizedSearchCV.
5. Cross-validation.
6. Bias-variance effects of different regularisation strengths.
7. Regression model comparison.

Dataset:
    California Housing dataset from scikit-learn.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Create output directory
# ---------------------------------------------------------

OUTPUT_DIR = Path("output_evidence/w4d2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Load the California Housing dataset
# ---------------------------------------------------------

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Target name:", housing.target_names[0])


# ---------------------------------------------------------
# 3. Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------------------------------
# 4. Evaluation function
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Calculate standard regression evaluation metrics."""

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


# ---------------------------------------------------------
# 5. Baseline Linear Regression
# ---------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_metrics = evaluate_model(
    linear_model,
    X_test,
    y_test,
)

print("\nBaseline Linear Regression")
print("--------------------------")

for metric, value in linear_metrics.items():
    print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------
# 6. Ridge Regression with GridSearchCV
# ---------------------------------------------------------

ridge_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ]
)

ridge_param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100],
}

ridge_grid = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=ridge_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

ridge_grid.fit(X_train, y_train)

best_ridge = ridge_grid.best_estimator_

ridge_metrics = evaluate_model(
    best_ridge,
    X_test,
    y_test,
)

print("\nRidge Grid Search")
print("-----------------")
print("Best parameters:", ridge_grid.best_params_)

for metric, value in ridge_metrics.items():
    print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------
# 7. Lasso Regression with GridSearchCV
# ---------------------------------------------------------

lasso_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000)),
    ]
)

lasso_param_grid = {
    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
}

lasso_grid = GridSearchCV(
    estimator=lasso_pipeline,
    param_grid=lasso_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

lasso_grid.fit(X_train, y_train)

best_lasso = lasso_grid.best_estimator_

lasso_metrics = evaluate_model(
    best_lasso,
    X_test,
    y_test,
)

print("\nLasso Grid Search")
print("-----------------")
print("Best parameters:", lasso_grid.best_params_)

for metric, value in lasso_metrics.items():
    print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------
# 8. Ridge Regression with RandomizedSearchCV
# ---------------------------------------------------------

ridge_random = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ]
)

ridge_random_params = {
    "model__alpha": np.logspace(-3, 3, 100),
}

ridge_random_search = RandomizedSearchCV(
    estimator=ridge_random,
    param_distributions=ridge_random_params,
    n_iter=20,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

ridge_random_search.fit(X_train, y_train)

best_ridge_random = ridge_random_search.best_estimator_

ridge_random_metrics = evaluate_model(
    best_ridge_random,
    X_test,
    y_test,
)

print("\nRidge Randomized Search")
print("-----------------------")
print("Best parameters:", ridge_random_search.best_params_)

for metric, value in ridge_random_metrics.items():
    print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------
# 9. Lasso Regression with RandomizedSearchCV
# ---------------------------------------------------------

lasso_random = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000)),
    ]
)

lasso_random_params = {
    "model__alpha": np.logspace(-4, 1, 100),
}

lasso_random_search = RandomizedSearchCV(
    estimator=lasso_random,
    param_distributions=lasso_random_params,
    n_iter=20,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

lasso_random_search.fit(X_train, y_train)

best_lasso_random = lasso_random_search.best_estimator_

lasso_random_metrics = evaluate_model(
    best_lasso_random,
    X_test,
    y_test,
)

print("\nLasso Randomized Search")
print("-----------------------")
print("Best parameters:", lasso_random_search.best_params_)

for metric, value in lasso_random_metrics.items():
    print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------
# 10. Compare all models
# ---------------------------------------------------------

results = pd.DataFrame(
    [
        {
            "Model": "Linear Regression",
            **linear_metrics,
        },
        {
            "Model": "Ridge GridSearchCV",
            **ridge_metrics,
        },
        {
            "Model": "Lasso GridSearchCV",
            **lasso_metrics,
        },
        {
            "Model": "Ridge RandomizedSearchCV",
            **ridge_random_metrics,
        },
        {
            "Model": "Lasso RandomizedSearchCV",
            **lasso_random_metrics,
        },
    ]
)

print("\nModel Comparison")
print("----------------")

print(results.to_string(index=False))


# ---------------------------------------------------------
# 11. Save model comparison evidence
# ---------------------------------------------------------

results.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# 12. Save Grid Search results
# ---------------------------------------------------------

ridge_grid_results = pd.DataFrame(
    ridge_grid.cv_results_
)

ridge_grid_results[
    [
        "param_model__alpha",
        "mean_test_score",
        "rank_test_score",
    ]
].to_csv(
    OUTPUT_DIR / "ridge_grid_search_results.csv",
    index=False,
)


lasso_grid_results = pd.DataFrame(
    lasso_grid.cv_results_
)

lasso_grid_results[
    [
        "param_model__alpha",
        "mean_test_score",
        "rank_test_score",
    ]
].to_csv(
    OUTPUT_DIR / "lasso_grid_search_results.csv",
    index=False,
)


# ---------------------------------------------------------
# 13. Save Randomized Search results
# ---------------------------------------------------------

ridge_random_results = pd.DataFrame(
    ridge_random_search.cv_results_
)

ridge_random_results[
    [
        "param_model__alpha",
        "mean_test_score",
        "rank_test_score",
    ]
].to_csv(
    OUTPUT_DIR / "ridge_randomized_search_results.csv",
    index=False,
)


lasso_random_results = pd.DataFrame(
    lasso_random_search.cv_results_
)

lasso_random_results[
    [
        "param_model__alpha",
        "mean_test_score",
        "rank_test_score",
    ]
].to_csv(
    OUTPUT_DIR / "lasso_randomized_search_results.csv",
    index=False,
)


# ---------------------------------------------------------
# 14. Visualise model comparison
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["R2"],
)

plt.ylabel("R² Score")
plt.xlabel("Model")
plt.title("W4D2: Model Performance Comparison")

plt.xticks(
    rotation=25,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison_r2.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 15. Demonstrate Bias-Variance Tradeoff
# ---------------------------------------------------------

# Different alpha values control the strength of
# Ridge regularisation.
#
# Smaller alpha:
#     Weaker regularisation and potentially higher variance.
#
# Larger alpha:
#     Stronger regularisation and potentially higher bias.

alpha_values = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
]

bias_variance_results = []


for alpha in alpha_values:

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=-1,
    )

    train_mse = -cv_results["train_score"].mean()
    validation_mse = -cv_results["test_score"].mean()

    bias_variance_results.append(
        {
            "alpha": alpha,
            "Training_MSE": train_mse,
            "Validation_MSE": validation_mse,
        }
    )


bias_variance_df = pd.DataFrame(
    bias_variance_results
)


print("\nBias-Variance Tradeoff")
print("----------------------")

print(
    bias_variance_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 16. Save Bias-Variance numerical evidence
# ---------------------------------------------------------

bias_variance_df.to_csv(
    OUTPUT_DIR / "bias_variance_results.csv",
    index=False,
)


# ---------------------------------------------------------
# 17. Plot Bias-Variance Tradeoff
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    bias_variance_df["alpha"],
    bias_variance_df["Training_MSE"],
    marker="o",
    label="Training MSE",
)

plt.plot(
    bias_variance_df["alpha"],
    bias_variance_df["Validation_MSE"],
    marker="o",
    label="Validation MSE",
)

plt.xscale("log")

plt.xlabel("Regularisation Strength (alpha)")
plt.ylabel("Mean Squared Error")

plt.title(
    "Bias-Variance Tradeoff with Ridge Regularisation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "bias_variance_tradeoff.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 18. Final summary
# ---------------------------------------------------------

print("\nEvidence saved to:", OUTPUT_DIR)

print("W4D2 execution completed successfully.")