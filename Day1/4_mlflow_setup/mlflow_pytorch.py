# import mlflow
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data

# # Create a simple PyTorch model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(10, 10)
#         self.fc2 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

# # Set up a machine learning experiment
# with mlflow.start_run() as run:
#     # Log some parameters
#     mlflow.log_param("learning_rate", 0.01)
#     mlflow.log_param("batch_size", 32)

#     # Set up the model, loss function, and optimizer
#     model = Net()
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)

#     # Train the model
#     for epoch in range(10):
#         for inputs, labels in data.DataLoader(...):
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # Log some metrics
#             mlflow.log_metric("loss", loss.item())
#             mlflow.log_metric("accuracy", ...)

#     # Log the model artifact
#     mlflow.pytorch.log_model(model, "models")
