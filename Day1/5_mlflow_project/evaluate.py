import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
from sklearn import datasets
import numpy as np
from torchmetrics.functional import accuracy
import pickle

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Evaluate():

    def __init__(self, args):
        self.args = args

        # prepare evauation data
        x_numpy, y_numpy = datasets.make_regression(n_samples=80, n_features=1, noise=9.7, random_state=2)
        X = torch.from_numpy(x_numpy.astype(np.float32))
        y = torch.from_numpy(y_numpy.astype(np.float32))
        y = y.view(y.shape[0], 1)

        # Set up a machine learning experiment
        # with mlflow.start_run() as run:

        # Set up the model and loss function
        model = Net()

        model = model.eval()
        state = torch.load(self.args.model_path)
        model.load_state_dict(state)

        criterion = nn.MSELoss()
        # evaluate the model
        outputs = model(X)
        loss = criterion(outputs, y) 
        # Log some metrics
        mlflow.log_metric("loss", loss.item())

                
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000/")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", default="model.pkl")
    
    args = parser.parse_args()

    # model evaluation
    model_evaluation = Evaluate
    model_evaluation(args=args)

