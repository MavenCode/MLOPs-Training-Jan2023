import argparse

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
import numpy as np

# prepare data

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1)
X = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_sample, n_features = X.shape


class Mymodel(mlflow.pyfunc.PythonModel):

    def __init__(self) -> None:
        super().__init__()

    def predict(self, context, model_input):
        return super().predict(context, model_input)

# Create a simple PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Train():

    def __init__(self, args, model):
        self.args = args
        self.model = model
        # Set up a machine learning experiment

        # log parameter
        mlflow.log_param("learning_rate", float(args.learning_rate))
        mlflow.log_param("epochs", args.epochs)
    
        # Set up the model, loss function, and optimizer
        model = self.model
        model.train()

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), self.args.learning_rate)

        # Train the model
        for epoch in range(self.args.epochs):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y) 
                loss.backward()
                optimizer.step()
                # Log some metrics
                mlflow.log_metric("loss", loss.item())
                
                if (epoch +1) % 5 == 0:
                    print(f"epoch {epoch+1}/{self.args.epochs}, loss: {loss.item()}")

            
           

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000/")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", default="Mlmodel")
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--epochs", default=100)

    args = parser.parse_args()

    with mlflow.start_run() as run:

        # model training
        model = Net()
        Train(args=args, model=model)

        # Log the model artifact
        model_info = mlflow.pytorch.log_model(model, args.model_path)

        # convert model artifact to a pyfunc model
        python_model = Mymodel()
        model_info = mlflow.pyfunc.log_model(artifact_path=model_info.artifact_path, python_model=python_model)
                
        # load the pyfunc model
        pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

        # inferencing 

        pyfunc_model.predict([])
