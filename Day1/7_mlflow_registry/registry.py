import mlflow
import argparse
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository


class DemoModel(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, context, model_input):
        return super().predict(context, model_input)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.MlflowClient()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="demo_model")
    args = parser.parse_args()

    with mlflow.start_run() as run:
        
        # create model
        model = DemoModel()

        # log the model
        mlflow.pyfunc.log_model(args.model_name, python_model=model)

        # register model in model registry
        try:
            client.create_registered_model(name=args.model_name)
        except mlflow.exceptions.RestException:     
            # The model is already registered
            pass

        run_uri = f"runs:/{run.info.run_id}/{args.model_name}"
        model_src = RunsArtifactRepository.get_underlying_uri(runs_uri=run_uri)
        model_desc = "pyfunc model demontration mlflow's Model Registry"

        # create a version of our model in the registry
        
        registered_model = client.create_model_version(
            name=args.model_name,
            description=model_desc,
            run_id=run.info.run_id,
            source=model_src,
        )

        # transition model to staging
        client.transition_model_version_stage(
            name=args.model_name,
            stage="staging",
            version=registered_model.version
        )

        print(f"registered model name: {registered_model.name}")
        print(f"registered model version: {registered_model.version}")
        print(f"registered model source: {registered_model.source}")
    


