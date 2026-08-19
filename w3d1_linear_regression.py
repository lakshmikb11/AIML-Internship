"""
W3D1: Linear Regression — Scikit-Learn

Tasks:
1. Train LinearRegression on a real-world regression dataset.
2. Evaluate using MSE, RMSE, MAE, and R².
3. Plot predicted vs actual values and residuals.
4. Train Ridge and Lasso models.
5. Compare all three models in a results table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

OUTPUT_DIR = Path("output_evidence") / "w3d1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------------------

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Target:", housing.target_names[0])
print("\nFeatures:")
print(list(X.columns))


# -------------------------------------------------------------------
# 2. Train-test split
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -------------------------------------------------------------------
# 3. Feature scaling
# -------------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------------------------
# 4. Define models
# -------------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=10000),
}


# -------------------------------------------------------------------
# 5. Train, predict, and evaluate models
# -------------------------------------------------------------------

results = []
predictions = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    predictions[model_name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append(
        {
            "Model": model_name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }
    )

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R²   : {r2:.6f}")

    # Print coefficients as required by the task.
    print("Coefficients:")
    for feature, coefficient in zip(X.columns, model.coef_):
        print(f"  {feature}: {coefficient:.6f}")

    print(f"Intercept: {model.intercept_:.6f}")


# -------------------------------------------------------------------
# 6. Save comparison results
# -------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)


# -------------------------------------------------------------------
# 7. Predicted vs Actual plot — Linear Regression
# -------------------------------------------------------------------

linear_predictions = predictions["Linear Regression"]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, linear_predictions, alpha=0.5)

min_value = min(y_test.min(), linear_predictions.min())
max_value = max(y_test.max(), linear_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predicted vs Actual")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "predicted_vs_actual.png",
    dpi=300,
)

plt.close()


# -------------------------------------------------------------------
# 8. Residual plot — Linear Regression
# -------------------------------------------------------------------

residuals = y_test - linear_predictions

plt.figure(figsize=(8, 6))
plt.scatter(linear_predictions, residuals, alpha=0.5)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residual Plot")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residuals.png",
    dpi=300,
)

plt.close()


# -------------------------------------------------------------------
# 9. Save sample predictions and residuals
# -------------------------------------------------------------------

prediction_evidence = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": linear_predictions,
        "Residual": residuals.values,
    }
)

prediction_evidence.to_csv(
    OUTPUT_DIR / "prediction_residuals.csv",
    index=False,
)


print("\nW3D1 completed successfully.")
print(f"Evidence saved in: {OUTPUT_DIR}")