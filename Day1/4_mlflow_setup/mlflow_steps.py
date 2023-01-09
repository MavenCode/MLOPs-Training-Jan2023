import mlflow
import logging

# Create an experiment
experiment_id = mlflow.create_experiment("my-experiment")
logging.info(f"Experiment ID: {experiment_id}")

# Start a new run
with mlflow.start_run(experiment_id=experiment_id, run_name="my-run") as run:
    # Log some parameters and metrics
    mlflow.log_param("param_1", 42)
    mlflow.log_metric("accuracy", 0.95)
    # End the run
    mlflow.end_run()