## MNIST example with MLFlow

In this example, we train a simple Pytorch Lightning model on the popular Mnist classification dataset.

this example illustrates the use of mlflow's api. it was specifically used in this example to:
*   create an experiment
*   create an experiment run
*   start an experiment run
*   log model parameters and matrics
*   log model and state dict as artifacts
*   create a registered model in our model registry
*   create a version of our registered model 

This example uses a postgres database running on docker as a backend store, artifacts are stored locally on the file system.
mlflow supports major databses like mysql, postgres and sqlite. for backend storage as well as major cloud storage providers like azure blob for artifact storage. 

The `model name`, `experiment name` and `mlflow server` info are decleared as environmental variables in the `Day4/mlflow_project/conda.yaml` file.  

## Mlflow Tracking server

You can start up an mlflow tracking server locally by running the command 
```
mlflow server
```

This will start an mlflow server on `localhost` port `5000`. you can specify a different port for the server with the `--port` option. 

For This section of the training, a custom mlflow tracking server with minio as the backend store has been provided. you can check out the server by navigating to [http://mlflow.cloudtraining-mavencode.com:5000](http://mlflow.cloudtraining-mavencode.com:5000). 

## Running the Project

To run the example via MLflow, navigate to the `Day4/Mlflow_project` directory and run the command

```
mlflow run .
```

This will run `main.py` with the default set of parameters such as `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where `X` is your desired value for `max_epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--env-manager=local`.

```
mlflow run . --env-manager=local
```

## Viewing results in the MLflow UI

You can view the run's metrics, parameters, and details by navigating to to the mlflow tracking server
[http://mlflow.cloudtraining-mavencode.com:5000](http://mlflow.cloudtraining-mavencode.com:5000).

On the Experiments side bar, click on the "Mnist_classification_experiment" and click on the latest run. you can also compare multiple runs accross various parameters and metrics

To checkout the models saved in the models registry, click on the "Models" tab on the ui page.  

For more details on MLflow tracking, see [the docs](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking).

