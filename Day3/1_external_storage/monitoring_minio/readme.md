# Industrial Equipment Monitoring Pipeline

This example showcases a complete machine learning workflow on Kubeflow, starting from `data ingestion` to `model deployment`. The pipeline ingests data from an external data store, `MinIO`, performs `preprocessing` and Principle Component Analysis `PCA` on the dataset, and trains a machine learning model using the preprocessed data. The final step of the workflow involves uploading the trained machine learning model to the `MinIO` server.

`Note`: In order to provide a simple example, this pipeline does not cover all the aspects of an industrial equipment monitoring system. The purpose of this example is to demonstrate the use of Kubeflow in a real-world scenario.

## Running The Pipeline

Run the following steps to creata a kubeflow pipeline run

### Compiling the pipeline

To run the pipeline, first we'll need to compile the pipeline into a `YAML` specification file using `kfp.Dsl`.to do this, navigate to the pipeline directory at `Day3/1_external_storage/monitoring_minio` and run the command 
```
python3 iot_monitoring.py
```

This will compile the `YAML` file for our pipeline, which can be uploaded to the kubeflow pipeline ui.

For instructions on how to upload a pipeline YAML file to `Kubeflow` and create a `pipeline run`, please refer to the `MLOpsTraining-Dec2022/README.md` file.

## TensorBoard

This guide outlines the steps for utilizing Tensorboard to visualize metrics of our model training pipeline step. The process begins by launching a Tensorboard server on Kubeflow, followed by connecting to the persistent volume claim associated `PVC` with our pipeline run. Finally, the `Tensorboard logs` directory will be mounted on the server, allowing you to access and analyze the relevant information."

### Selecting the proper PVC

If the wrong volume is attached to our TensorBoard server, we woulnd be able to visualize our metrics. 

![tb_0](https://user-images.githubusercontent.com/97686519/217696100-cab8d6b6-8ec1-4798-91c3-f8139fe1fc4b.png)

### Create a new Tensorboard server

To access the Tensorboard in the Kubeflow Dashboard, follow these steps:

* Click on the "Tensorboard" option in the Kubeflow Dashboard.
* Select the "New Tensorboard" option to initiate the process of creating a new Tensorboard instance. "

![tb_1](https://user-images.githubusercontent.com/97686519/217696295-6e6054ed-0c46-4ba6-99d1-65a5040d534b.png)

To create a Tensorboard server, perform the following actions:

* Enter a descriptive name for the Tensorboard server in the designated field.
* Click on the "PVC" option and select the PVC created from the Kubeflow pipeline run.
* Specify the mount path as "tensorboard/logs".
* Click on the "Create" button to finalize the creation of the Tensorboard server."

![tb_2](https://user-images.githubusercontent.com/97686519/217696314-9b9804f2-ab0d-439f-b273-02345da1b4d7.png)

### Connecting to our server

Once the server creation is complete, click on `CONNECT` to open the `Tensorboard Dashboard`.

![tb_3](https://user-images.githubusercontent.com/97686519/217696500-cdfb3f67-c15e-4aca-b73e-8d3b89268b13.png)

![tb_4](https://user-images.githubusercontent.com/97686519/217696506-76663898-1c9c-492b-9b9d-9ebbb60a5ed5.png)
