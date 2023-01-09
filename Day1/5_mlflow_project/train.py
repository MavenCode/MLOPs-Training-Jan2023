import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import argparse
from sklearn import datasets
import numpy as np
from torchmetrics.functional import accuracy

# prepare data

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1)
X = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_sample, n_features = X.shape

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

    def __init__(self, args):
        self.args = args
        # Set up a machine learning experiment

        # log parameter
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("epochs", args.epochs)
    
        # Set up the model, loss function, and optimizer
        model = Net()
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
        
        mlflow.pytorch.save_state_dict(state_dict=model.state_dict(), path=self.args.model_path)
        # Log the model artifact
        mlflow.pytorch.log_model(model, self.args.model_path)
        mlflow.end_run()

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000/")
    client = mlflow.MlflowClient()
    
    try:
        experiment_id = client.create_experiment("Mlflow Demo")
    except mlflow.exceptions.RestException:
        experiment_id = client.get_experiment_by_name("Mlflow Demo").experiment_id

    # Create an MLFlow run using the mlflow client

    run_name = f"mnist_run_{datetime.datetime.now()}"    
    run = client.create_run(experiment_id=experiment_id, run_name=run_name)

    ml_run = mlflow.start_run(run_id=run.info.run_id)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./model.pt", type=str)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--epochs", default=100, type=int)

    args = parser.parse_args()

    # model training
    model_train = Train
    model_train(args=args)

