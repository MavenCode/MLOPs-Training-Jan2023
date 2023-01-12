import pytorch_lightning as pl
import os
import mlflow
from datetime import datetime
import mlflow.pytorch
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository



from model import MnistModel

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    args = parser.parse_args()

    experiment_name = os.getenv("MLFLOW_EXPERIMENT")
    model_name = os.getenv("MODEL_NAME")

    client = mlflow.MlflowClient()

    # Create Mlflow Experiment 

    try:
        experiment_id = client.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    # Create an MLFlow run using the mlflow client

    run_name = f"mnist_run_{datetime.now()}"    
    run = client.create_run(experiment_id=experiment_id, run_name=run_name)
    mlflow_run = mlflow.start_run(run_id=run.info.run_id)

    # Pytorch lightning Model

    mnist_model = MnistModel(kwargs=args)
    
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
        accelerator=args.accelerator, 
        devices=args.devices, 
        num_nodes=args.num_nodes)
    
    # Enable auto logging. log_model is set to false since we're going to add the model to the model registry using the mlflow client 

    mlflow.pytorch.autolog(log_models=False)

    # Train and Test model

    trainer.fit(mnist_model)
    trainer.test(mnist_model)

    # logging pytorch model and model's state dict. this wwould be stored as mlflow artifacts

    state_dict = mnist_model.state_dict()

    mlflow.pytorch.log_model(mnist_model, artifact_path=f"{model_name}")
    mlflow.pytorch.log_state_dict(state_dict=state_dict, artifact_path="mnist_model_state_dict")


    model_desc = "pytorch mnist classification model"
    runs_uri = f"runs:/{mlflow_run.info.run_id}/{model_name}"
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)

    # register model in model registry
    try:
        client.create_registered_model(name=model_name)
    except mlflow.exceptions.RestException:     
        # The model is already registered
        pass

    # create a version of our model in the registry

    client.create_model_version(
        name=model_name,
        description=model_desc,
        run_id=mlflow_run.info.run_id,
        source=model_src,
    )

    mlflow.end_run()
