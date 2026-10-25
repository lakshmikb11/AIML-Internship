
"""W11D1: MLflow experiment tracking and model registry."""

from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configuration
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "W11D1_MLflow_Experiment_Tracking"
MODEL_NAME = "w11d1_best_diabetes_model"
EVIDENCE_DIR = Path("output_evidence") / "w11d1"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Configure MLflow tracking with a database-backed model registry.
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load a built-in dataset; no external download is needed.
dataset = load_diabetes(as_frame=True)
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Five experiments with different hyperparameter configurations.
experiments = [
    {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 150, "max_depth": 8, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 4},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
]

results = []

print("=" * 65)
print("W11D1: MLFLOW EXPERIMENT TRACKING")
print("=" * 65)
print(f"MLflow version: {mlflow.__version__}")
print(f"Tracking URI: {TRACKING_URI}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of experiments: {len(experiments)}")

# Run and log all five experiments.
for index, params in enumerate(experiments, start=1):
    with mlflow.start_run(run_name=f"experiment_{index}") as run:
        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        # Log parameters, metrics, and the trained model artifact.
        mlflow.log_params(params)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("training_samples", len(X_train))
        mlflow.log_metric("testing_samples", len(X_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.iloc[:3],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        results.append(
            {
                "run_id": run.info.run_id,
                "params": params.copy(),
                "rmse": rmse,
                "r2_score": r2,
            }
        )

        print(
            f"\nExperiment {index}/5"
            f"\n  Parameters: {params}"
            f"\n  RMSE: {rmse:.4f}"
            f"\n  R2 score: {r2:.4f}"
            f"\n  Run ID: {run.info.run_id}"
        )

# Select the best model by highest R2 score.
best_result = max(results, key=lambda item: item["r2_score"])
best_run_id = best_result["run_id"]
best_model_uri = f"runs:/{best_run_id}/model"

print("\n" + "=" * 65)
print("REGISTERING THE BEST MODEL")
print("=" * 65)
print(f"Best run ID: {best_run_id}")
print(f"Best parameters: {best_result['params']}")
print(f"Best R2 score: {best_result['r2_score']:.4f}")
print(f"Best RMSE: {best_result['rmse']:.4f}")

# Register the winning run's model artifact.
client = MlflowClient()

registered_model = mlflow.register_model(
    model_uri=best_model_uri,
    name=MODEL_NAME,
)

model_version = registered_model.version
registered_model_uri = f"models:/{MODEL_NAME}/{model_version}"

# Load the registered version and verify its predictions.
loaded_model = mlflow.sklearn.load_model(registered_model_uri)
registered_predictions = loaded_model.predict(X_test)

registered_r2 = float(r2_score(y_test, registered_predictions))
registered_rmse = float(
    np.sqrt(mean_squared_error(y_test, registered_predictions))
)

if not np.all(np.isfinite(registered_predictions)):
    raise RuntimeError("Registered model produced non-finite predictions.")

if not np.isclose(registered_r2, best_result["r2_score"]):
    raise RuntimeError("Registered model score does not match the best run.")

print(f"\nRegistered model name: {MODEL_NAME}")
print(f"Registered model version: {model_version}")
print(f"Registered model URI: {registered_model_uri}")
print(f"Loaded model R2 score: {registered_r2:.4f}")
print(f"Loaded model RMSE: {registered_rmse:.4f}")
print(f"Sample predictions: {registered_predictions[:5].round(2).tolist()}")

# Save reproducible output evidence.
evidence_path = EVIDENCE_DIR / "w11d1_experiment_results.txt"

with evidence_path.open("w", encoding="utf-8") as evidence:
    evidence.write("W11D1: MLflow Experiment Tracking\n")
    evidence.write("=" * 55 + "\n")
    evidence.write(f"MLflow version: {mlflow.__version__}\n")
    evidence.write(f"Tracking URI: {TRACKING_URI}\n")
    evidence.write(f"Experiment: {EXPERIMENT_NAME}\n")
    evidence.write(f"Model registry name: {MODEL_NAME}\n\n")

    for index, result in enumerate(results, start=1):
        evidence.write(f"Experiment {index}\n")
        evidence.write(f"  Run ID: {result['run_id']}\n")
        evidence.write(f"  Parameters: {result['params']}\n")
        evidence.write(f"  RMSE: {result['rmse']:.4f}\n")
        evidence.write(f"  R2 score: {result['r2_score']:.4f}\n\n")

    evidence.write("BEST MODEL\n")
    evidence.write(f"Best run ID: {best_run_id}\n")
    evidence.write(f"Best parameters: {best_result['params']}\n")
    evidence.write(f"Registered version: {model_version}\n")
    evidence.write(f"Registered URI: {registered_model_uri}\n")
    evidence.write(f"Reloaded R2 score: {registered_r2:.4f}\n")
    evidence.write(f"Reloaded RMSE: {registered_rmse:.4f}\n")
    evidence.write(
        f"Sample predictions: {registered_predictions[:5].round(2).tolist()}\n"
    )
    evidence.write("\nSTATUS: All five experiments completed successfully.\n")
    evidence.write("STATUS: Best model registered and reloaded successfully.\n")

print(f"\nEvidence saved to: {evidence_path}")
print("\nW11D1 experiment tracking and registration completed.")
