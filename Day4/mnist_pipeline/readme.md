## Mnist Pipeline with Mlflow

in this example, we took the `mlflow project` example from `Day4/mlflow_project` and converted it into a kubeflow pipeline.

## componets
this is a very simple pipeline that goes to further explain the use of mlflow alongside kubeflow.

the components for this pipeline are 
* pvc component to create a persistent volume
* mlflow_init component to create an experiment
* data prep component to download and prep the Mnist Dataset
* train component to carry out model training
* test to test our model

similar to the `Day4/iot_mlflow_pipeline`, we're creating an `mlflow experiment` for our kubeflow pipeline run and the result of our experiment can be viewed 
at the mlflow server at [http://mlflow.cloudtraining-mavencode.com:5000](http://mlflow.cloudtraining-mavencode.com:5000).