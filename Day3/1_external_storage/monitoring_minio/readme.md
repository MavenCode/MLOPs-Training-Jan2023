### Pipeline for Industrial Equipment Monitoring

## Prerequisites

1. configure minio client -> `wget https://dl.min.io/client/mc/release/linux-amd64/mc && chmod +x mc`
2. validate minio client is working `./mc --help`
3. connect to minio server -> `mc alias set minio http://minio-service.kubeflow:9000 minio minio123`
4. validate needed data is available -> 

## How to Implement Kubeflow Pipelines Components
In this pipeline, we have the following components:
1. Sensor data download component
2. Preprocess the dataset component
3. Train the model component
4. Test model component
5. Print Results component
6. Export the trained model component