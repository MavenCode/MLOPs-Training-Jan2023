# MLflow Setup

## Installations
To set up local MLflow runs, you will need to install the MLflow library and create an experiment.

### Prepare Python Environment (Anaconda):
Create Python Environment in Home Folder:
```
conda create --name myenv
```
Activate Python Environment:
```
conda activate myenv
```

Install mlflow and pytorch within Anaconda:
```
conda install -c conda-forge mlflow 
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Here is an example of how to set up local MLflow runs:

1. Create an Experiment:
```
import mlflow

# Create an experiment
experiment_id = mlflow.create_experiment("my-experiment")
print("Experiment ID:", experiment_id)

```

2. Start new Run:
```
import mlflow
# Start a new run
with mlflow.start_run(experiment_id=experiment_id, run_name="my-run") as run:
    # Log some parameters and metrics
    mlflow.log_param("param_1", 42)
    mlflow.log_metric("accuracy", 0.95)
    # End the run
    mlflow.end_run()
```

By default, MLflow will write the run data to a local file in the mlruns directory. You can specify a different directory by setting the `MLFLOW_TRACKING_URI` environment variable to a file:// URI.

For example:
```export MLFLOW_TRACKING_URI=file:///path/to/mlruns```

You can then use the MLflow Tracking API or the MLflow user interface to access and query the run data.

## Access MLflow
To access the MLflow user interface (UI) from a local install, you will need to start the MLflow server and point your web browser to the correct URL.

Here is an example of how to access the MLflow UI from a local install:

1. Start the MLflow server:
```
mlflow server
```

2. Open a web browser and navigate to the MLflow UI at the following URL:
```
http://localhost:5000
```
By default, the MLflow server will listen on port 5000. If you have configured MLflow to listen on a different port,
you will need to use the correct port number in the URL.

Once you have accessed the MLflow UI, you should be able to browse and query the data for your machine learning experiments.

## Labs
we have put together some practical examples demonstrating simple usage of mlflow in your machine learning projects. Integrating Mlflow into machine learning projects is simple and fast. this section shows you how to achieve this.

### Experiments and runs

The `pytorch_steps` shows how to create an mlflow experiment using the mlflow api and starting and ending an mlflow run for that experiment. it also demonstrates `mlflow tracking` to log parameters and matrics. 

### Tracking server

In the `mlflow_steps_tracking_server` example, we carried out all the operations from the first example as well as setting a default tracking servers for our experiment. this is very useful when you have multiple servers or your server is hosted on the web. the results of our experiments will be routed to the tracking server we set. 

### Connecting to a database store

The `mlflow_steps_sqlite` example demonstrates connecting mlflow to a backend database store. this is needed when using mlflow's `Model` and `Model Registry` features to log, track, version and serve machine learning models

### Mlflow with Pytorch

In the `mlflow_pytorch` example, we demonstrated the use of mlflow's api in a simple pytorch training code. mlflow is easy to integrate into projects and we can setup mlflow tracking in our project with just a couple of lines of code. in this example, we used the mlflow api to start an mlflow run with `mlflow.start_run()`, we logged model parameters, in this case the batch size and learning rate with `mlflow.log_param()`, we also logged the training loss and training accuracy of our model with `mlflow.log_metrics()`. finally, using the mlflow's pytorch api, we logged the trained machine learning model as an artifact with `mlflow.pytorch.log_model()`. 

This example although being pytorch specific, shows the basis of integrating mlflow into any machine learning project.



