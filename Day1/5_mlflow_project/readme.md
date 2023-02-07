# MLflow Project

This shows a quick guide on `Mlflow project` configuration. 

## Mlprojects

The `Mlproject` file contains the configuration for our Mlflow project. Every Mlflow project must have an `Mlproject` file. This file contains the specifications for our development environment, dependencies and requirements and the entry points of our projects

## Environment

Mlflow projects are typically ran within a virtual environment although this can be bypasses. the `environment` yaml file consist of the specifications for our virtual environment. in this example, we're using a `conda` environment but other options like `pip env` can be used.

## Project Packages and Requirements

we can specify the pip requirements for our project on a separate `requirments` text file or specify them in our `environment` file. 

## Training and evaluation 

The `MlProject` file contains the entry point for our project, in this example. we have `train` and `evaluate` python scripts. this carries model training and evaluation of our machine learning model'

## Running the Project

To run the example via MLflow, navigate to the `Day1/5_mlflow_projeect/mlproject` directory and run the command

```
mlflow run .
```

This will run the project with the default parameters. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P epochs=X
```

where `X` is your desired value for `max_epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--env-manager=local`.

```
mlflow run . --env-manager=local
```
## Viewing results in the MLflow UI

Start up an mlflow server by running the command
```
mlflow server
```
this will start up a mlflow tracking server on `localhost:5000`. you can specify a different port by adding the `--port` option to the above command, followed by your desired port.

Once the code is finished executing, you can view the run's metrics, parameters, and details by navigating to to the server we set up earlier.
[http://localhost:5000](http://localhost:5000).


On the Experiments side bar, click on the "Mnist_classification_experiment" and click on the latest run. you can also compare multiple runs across various parameters and metrics

To checkout the models saved in the models registry, click on the "Models" tab on the ui page.  

For more details on MLflow tracking, see [the docs](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking).


