def mlflow_experiment(mlflow_server: str, experiment_name: str) -> str:
    import mlflow
    
    mlflow.set_tracking_uri(mlflow_server)

    client = mlflow.MlflowClient()

    try:
        experiment_id = client.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        # experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)

        experiment_id = experiment.experiment_id
        
    return experiment_id
