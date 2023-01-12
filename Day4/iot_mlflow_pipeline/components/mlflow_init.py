def create_experiment(experiment_name: str) -> str:
    
    """Create an mlflow experiment for this pipeline run"""
    import sys, subprocess;
    from datetime import datetime
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlflow'])

    import mlflow

    mlflow.set_tracking_uri("http://mlflow.cloudtraining-mavencode.com:5000")
    
    client = mlflow.MlflowClient()
    # Create Mlflow Experiment 
    try:
        experiment_id = client.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment_id=experiment.experiment_id)
        experiment_id = experiment.experiment_id

    return experiment_id
    