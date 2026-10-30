
import json
import mlflow

# --------------------------------------------------
# 1. Load existing model evaluation results
# --------------------------------------------------

with open("outputs/model_evaluation.json", "r") as file:
    results = json.load(file)

# --------------------------------------------------
# 2. Set MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("Real_Estate_Investment_Advisor")

# --------------------------------------------------
# 3. Start an MLflow run
# --------------------------------------------------

with mlflow.start_run(run_name="ExtraTrees_Model_Evaluation"):

    # Log model information
    mlflow.log_param("model_name", "ExtraTreesRegressor")
    mlflow.log_param("dataset", "India Housing Prices")
    mlflow.log_param("task", "Property Price Prediction")

    # Log trained model metrics
    mlflow.log_metric(
        "trained_model_MAE",
        results["Trained_Model"]["MAE"]
    )

    mlflow.log_metric(
        "trained_model_RMSE",
        results["Trained_Model"]["RMSE"]
    )

    mlflow.log_metric(
        "trained_model_R2",
        results["Trained_Model"]["R2"]
    )

    # Log baseline metrics
    mlflow.log_metric(
        "baseline_MAE",
        results["Mean_Baseline"]["MAE"]
    )

    mlflow.log_metric(
        "baseline_RMSE",
        results["Mean_Baseline"]["RMSE"]
    )

    mlflow.log_metric(
        "baseline_R2",
        results["Mean_Baseline"]["R2"]
    )

    print("MLflow experiment logged successfully!")

print("Model evaluation tracking completed.")
