
"""
W4D3: Model Serialisation — joblib & pickle

This script demonstrates:
1. Training a Linear Regression model.
2. Evaluating the original trained model.
3. Serialising the trained model using joblib.
4. Serialising the trained model using pickle.
5. Loading both serialised models.
6. Comparing predictions from the original and restored models.
7. Verifying that model serialisation preserves prediction behaviour.

Dataset:
    California Housing dataset from scikit-learn.
"""

from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR = Path("output_evidence/w4d3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBLIB_PATH = OUTPUT_DIR / "linear_regression_model.joblib"
PICKLE_PATH = OUTPUT_DIR / "linear_regression_model.pkl"


# ---------------------------------------------------------
# 2. Load dataset
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
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------------------------------
# 4. Train Linear Regression model
# ---------------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. Evaluate original model
# ---------------------------------------------------------

original_predictions = model.predict(X_test)

mse = mean_squared_error(y_test, original_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, original_predictions)
r2 = r2_score(y_test, original_predictions)

print("\nOriginal Model Evaluation")
print("-------------------------")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"R2:   {r2:.6f}")


# ---------------------------------------------------------
# 6. Serialise using joblib
# ---------------------------------------------------------

joblib.dump(model, JOBLIB_PATH)

print("\nJoblib Serialization")
print("--------------------")
print("Model saved to:", JOBLIB_PATH)


# ---------------------------------------------------------
# 7. Serialise using pickle
# ---------------------------------------------------------

with open(PICKLE_PATH, "wb") as pickle_file:
    pickle.dump(model, pickle_file)

print("\nPickle Serialization")
print("--------------------")
print("Model saved to:", PICKLE_PATH)


# ---------------------------------------------------------
# 8. Load model using joblib
# ---------------------------------------------------------

joblib_model = joblib.load(JOBLIB_PATH)

joblib_predictions = joblib_model.predict(X_test)


# ---------------------------------------------------------
# 9. Load model using pickle
# ---------------------------------------------------------

with open(PICKLE_PATH, "rb") as pickle_file:
    pickle_model = pickle.load(pickle_file)

pickle_predictions = pickle_model.predict(X_test)


# ---------------------------------------------------------
# 10. Verify predictions
# ---------------------------------------------------------

joblib_match = np.allclose(
    original_predictions,
    joblib_predictions,
)

pickle_match = np.allclose(
    original_predictions,
    pickle_predictions,
)

print("\nSerialization Verification")
print("--------------------------")
print("Joblib predictions match original:", joblib_match)
print("Pickle predictions match original:", pickle_match)


# ---------------------------------------------------------
# 11. Evaluate restored models
# ---------------------------------------------------------

def calculate_metrics(target, predictions):
    """Calculate standard regression metrics."""

    mse_value = mean_squared_error(target, predictions)

    return {
        "MSE": mse_value,
        "RMSE": np.sqrt(mse_value),
        "MAE": mean_absolute_error(target, predictions),
        "R2": r2_score(target, predictions),
    }


joblib_metrics = calculate_metrics(
    y_test,
    joblib_predictions,
)

pickle_metrics = calculate_metrics(
    y_test,
    pickle_predictions,
)


# ---------------------------------------------------------
# 12. Compare original and restored models
# ---------------------------------------------------------

comparison = pd.DataFrame(
    [
        {
            "Model": "Original Linear Regression",
            **calculate_metrics(
                y_test,
                original_predictions,
            ),
        },
        {
            "Model": "Joblib Restored Model",
            **joblib_metrics,
        },
        {
            "Model": "Pickle Restored Model",
            **pickle_metrics,
        },
    ]
)

print("\nModel Comparison")
print("----------------")
print(comparison.to_string(index=False))


# ---------------------------------------------------------
# 13. Save comparison evidence
# ---------------------------------------------------------

comparison.to_csv(
    OUTPUT_DIR / "serialization_model_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# 14. Save verification evidence
# ---------------------------------------------------------

verification = pd.DataFrame(
    [
        {
            "Serialization_Method": "joblib",
            "File": str(JOBLIB_PATH),
            "File_Exists": JOBLIB_PATH.exists(),
            "Predictions_Match": joblib_match,
        },
        {
            "Serialization_Method": "pickle",
            "File": str(PICKLE_PATH),
            "File_Exists": PICKLE_PATH.exists(),
            "Predictions_Match": pickle_match,
        },
    ]
)

verification.to_csv(
    OUTPUT_DIR / "serialization_verification.csv",
    index=False,
)


# ---------------------------------------------------------
# 15. Save sample prediction evidence
# ---------------------------------------------------------

prediction_evidence = pd.DataFrame(
    {
        "Actual": y_test.iloc[:10].to_numpy(),
        "Original_Prediction": original_predictions[:10],
        "Joblib_Prediction": joblib_predictions[:10],
        "Pickle_Prediction": pickle_predictions[:10],
    }
)

prediction_evidence.to_csv(
    OUTPUT_DIR / "serialization_prediction_evidence.csv",
    index=False,
)


# ---------------------------------------------------------
# 16. Final verification
# ---------------------------------------------------------

if not JOBLIB_PATH.exists():
    raise FileNotFoundError(
        "Joblib model file was not created."
    )

if not PICKLE_PATH.exists():
    raise FileNotFoundError(
        "Pickle model file was not created."
    )

if not joblib_match:
    raise AssertionError(
        "Joblib predictions do not match original predictions."
    )

if not pickle_match:
    raise AssertionError(
        "Pickle predictions do not match original predictions."
    )

print("\nEvidence saved to:", OUTPUT_DIR)
print("W4D3 execution completed successfully.")

