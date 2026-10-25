
"""W11D2: MLflow Model Registry, versioning, and model serving."""

from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "W11D2_MLflow_Model_Registry"
MODEL_NAME = "w11d2_best_diabetes_model"
EVIDENCE_DIR = Path("output_evidence") / "w11d2"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------------------------------------
# Dataset and reproducible train/test split
# ---------------------------------------------------------
dataset = load_diabetes(as_frame=True)
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Five distinct hyperparameter configurations.
experiments = [
    {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 150, "max_depth": 8, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 4},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
]

results = []

print("=" * 65)
print("W11D2: MLFLOW MODEL REGISTRY AND VERSIONING")
print("=" * 65)
print(f"MLflow version: {mlflow.__version__}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Tracking URI: {TRACKING_URI}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Experiments to run: {len(experiments)}")

# ---------------------------------------------------------
# Train and log five experiments
# ---------------------------------------------------------
for index, params in enumerate(experiments, start=1):
    with mlflow.start_run(
        run_name=f"w11d2_random_forest_{index}"
    ) as run:
        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = float(
            np.sqrt(mean_squared_error(y_test, predictions))
        )
        r2 = float(r2_score(y_test, predictions))

        mlflow.log_params(params)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("dataset", "sklearn_diabetes")
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
                "run_name": f"w11d2_random_forest_{index}",
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "rmse": rmse,
                "r2_score": r2,
            }
        )

        print(f"\nExperiment {index}/5")
        print(f"Parameters: {params}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 score: {r2:.4f}")
        print(f"Run ID: {run.info.run_id}")

# ---------------------------------------------------------
# Compare experiments: lowest RMSE wins
# ---------------------------------------------------------
comparison = pd.DataFrame(results).sort_values(
    by=["rmse", "r2_score"],
    ascending=[True, False],
).reset_index(drop=True)

comparison.insert(
    0,
    "rank",
    range(1, len(comparison) + 1),
)

best_result = comparison.iloc[0].to_dict()
best_run_id = str(best_result["run_id"])
best_model_uri = f"runs:/{best_run_id}/model"

print("\n" + "=" * 65)
print("EXPERIMENT COMPARISON — BEST RMSE FIRST")
print("=" * 65)
print(
    comparison[
        [
            "rank",
            "run_name",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "rmse",
            "r2_score",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Register the best model
# ---------------------------------------------------------
client = MlflowClient()

registered_model = mlflow.register_model(
    model_uri=best_model_uri,
    name=MODEL_NAME,
)

model_version = str(registered_model.version)
registered_model_uri = f"models:/{MODEL_NAME}/{model_version}"

# Load the exact registered version and validate it.
loaded_model = mlflow.sklearn.load_model(registered_model_uri)
registered_predictions = loaded_model.predict(X_test)

registered_rmse = float(
    np.sqrt(mean_squared_error(y_test, registered_predictions))
)
registered_r2 = float(r2_score(y_test, registered_predictions))

if not np.all(np.isfinite(registered_predictions)):
    raise RuntimeError("Registered model returned invalid predictions.")

if not np.isclose(registered_rmse, float(best_result["rmse"])):
    raise RuntimeError("Registered model RMSE differs from its source run.")

if not np.isclose(registered_r2, float(best_result["r2_score"])):
    raise RuntimeError("Registered model R2 differs from its source run.")

print("\n" + "=" * 65)
print("MODEL REGISTRY VERIFICATION")
print("=" * 65)
print(f"Registered model: {MODEL_NAME}")
print(f"Model version: {model_version}")
print(f"Model URI: {registered_model_uri}")
print(f"Source run ID: {best_run_id}")
print(f"Verified RMSE: {registered_rmse:.4f}")
print(f"Verified R2 score: {registered_r2:.4f}")
print(f"Sample predictions: {registered_predictions[:5].round(2).tolist()}")

# ---------------------------------------------------------
# Save reproducible evidence
# ---------------------------------------------------------
comparison_path = EVIDENCE_DIR / "w11d2_experiment_comparison.csv"
results_path = EVIDENCE_DIR / "w11d2_model_registry_results.txt"
metadata_path = EVIDENCE_DIR / "w11d2_model_metadata.json"

comparison.to_csv(comparison_path, index=False)

metadata = {
    "experiment_name": EXPERIMENT_NAME,
    "model_name": MODEL_NAME,
    "model_version": model_version,
    "registered_model_uri": registered_model_uri,
    "best_run_id": best_run_id,
    "selection_metric": "rmse",
    "best_rmse": registered_rmse,
    "best_r2_score": registered_r2,
    "tracking_uri": TRACKING_URI,
}

with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

with results_path.open("w", encoding="utf-8") as file:
    file.write("W11D2: MLflow Model Registry and Versioning\n")
    file.write("=" * 55 + "\n")
    file.write(f"MLflow version: {mlflow.__version__}\n")
    file.write(f"Experiment: {EXPERIMENT_NAME}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"Completed experiments: {len(results)}\n\n")

    file.write("EXPERIMENT COMPARISON (BEST RMSE FIRST)\n")
    file.write(comparison.to_string(index=False))
    file.write("\n\nREGISTERED MODEL\n")
    file.write(f"Model name: {MODEL_NAME}\n")
    file.write(f"Version: {model_version}\n")
    file.write(f"URI: {registered_model_uri}\n")
    file.write(f"Source run ID: {best_run_id}\n")
    file.write(f"Verified RMSE: {registered_rmse:.4f}\n")
    file.write(f"Verified R2 score: {registered_r2:.4f}\n")
    file.write(
        f"Sample predictions: "
        f"{registered_predictions[:5].round(2).tolist()}\n"
    )
    file.write("\nSTATUS: Five experiments completed.\n")
    file.write("STATUS: Best model registered and reloaded.\n")
    file.write("STATUS: Registered predictions verified.\n")
    file.write(
        "\nMODEL SERVING COMMAND (run from repository root):\n"
    )
    file.write(
        f"mlflow models serve "
        f"--model-uri {registered_model_uri} "
        f"--host 127.0.0.1 --port 5001 --env-manager local\n"
    )

print("\nEvidence files saved:")
print(f"- {comparison_path}")
print(f"- {results_path}")
print(f"- {metadata_path}")
print("\nW11D2 experiment tracking and registry verification completed.")
print("\nOpen the MLflow UI to compare the five recorded runs.")
