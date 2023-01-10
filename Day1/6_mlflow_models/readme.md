# Mlflow Models

The Mlflow `Model` feature allows us to package machine learning models and their dependecies. It allows us to very quickly take our model from model training to serving

This example demonstrates the use of mlflow's `pyfunc` module to serve a machine learning model that was saved as artifact from an mlflow run.

To run this example, create and activate a conda environment by running the commands 
```
conda create -n mlflow python=3.10.8
conda activate mlflow
```
Next, install mlflow and pytorch
```
pip install mlflow pytorch
```
once that's done, you can run the python script with the command

```
python3 mlflow_pyfunc.py
```

