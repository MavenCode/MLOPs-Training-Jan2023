# Mlflow Registry

Mlflows Model Registry is a feature of mlflow that lets us properly version, stage and organize our machine learning models. it provides a centralized model store 
for storing and tracking of machine learning models. 

In this example, we demonstrate the use of mlflow's API to register a model to MLflow Registry, create a version of that model as well as to transistion the model to staging

To run this example, create and activate a conda environment by running the commands.
```
conda create -n mlflow python=3.10.8
conda activate mlflow
```
if you already have a conda environment, you can skip the creation and simply activate the environment.

Next, install mlflow 
```
pip install mlflow 
```
you can now run the example by running the command 
```
python3 registry.py
```

once the run is done, go to the mlflow tracking server and click on the models tab to preview the registered model. 
