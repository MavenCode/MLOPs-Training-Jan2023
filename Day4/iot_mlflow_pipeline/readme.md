## IOT Pipeline With Kubeflow

In this example, we're adding mlflow tracking to the iot pipeline from `Day3/Monitoring_pipeline` to track artifacts, models and parameters from our pipeline componenets. 
### Pipeline Components
The components in this pipeline is similar to the source pipeline. we still have the following components
* Sensor data download component
* Preprocess the dataset component
* carry our PCA
* Train the model component
* Test model component
* Print Results component

we have an `mlflow_init` component that creates an experiment for this the pipeline run. 
all components that require tracking or atifact logging are started as an `mlflow run` in our tracking server under the experiment we created in the `mlflow_init` commponent. 

You can specify the name of your experiment during the preocess of creating the kubeflow pipeline run on the kubeflow dashboard.

In the `Train` component, used `mlflow.keras` to log our model to mlflow and so we no longer needed the `export_model` component from the previous pipeline 

### results

The metrics, parameters and artifacts from this pipeline can be seen at the mlflow `tracking server` UI. 
we're using the custom tracking server provided for this section. you can view it by navigating to [http://mlflow.cloudtraining-mavencode.com:5000](http://mlflow.cloudtraining-mavencode.com:5000).