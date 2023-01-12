import os

import torch
from torch.nn import functional as F
from torch import nn
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import mlflow

class MnistModel(pl.LightningModule):
    def __init__(self, kwargs):
        super(MnistModel, self).__init__()

        self.kwargs = kwargs
        self.conv_layer1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(10, 20,kernel_size=5)
        self.conv_layer2_drop = nn.Dropout2d()

        self.layer_3 = nn.Linear(320,50)
        self.layer_4 = nn.Linear(50,10)


    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        if stage == 'fit' or stage is None:
            mnist_train = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_train, [55000,5000])
        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(os.getcwd(), train=False, download=True, transform=transform)

    def forward(self, x):
        output_conv1 = F.relu(F.max_pool2d(self.conv_layer1(x),2))
        output_conv2 = F.relu(F.max_pool2d(self.conv_layer2_drop(self.conv_layer2(output_conv1)),2))

        input_fc1 = output_conv2.view(-1,320)
        output_fc1 = F.relu(self.layer_3(input_fc1))
        output_dropout = F.dropout(output_fc1, training=self.training)
        output_layer = self.layer_4(output_dropout)
        return F.log_softmax(output_layer)

    def loss_function(self, output, target):

        return  nn.CrossEntropyLoss()(output.view(-1, 10),target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.kwargs.learning_rate)
        return optimizer


    def training_step(self, batch, batch_index):
        x, y = batch
        label = y.view(-1)
        out = self.forward(x)
        loss = self.loss_function(out, label)

        log = {'training_loss': loss}

        mlflow.log_metric("training_loss", loss)
        
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_index):
        x, y = batch
        label = y.view(-1)
        out = self(x)
        loss = self.loss_function(out, label)

        mlflow.log_metric("validation_loss", loss)

        return {'val_loss': loss}


    def test_step(self, batch, batch_index):
        x, y = batch
        out = self.forward(x)

        _, y_hat = torch.max(out, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu(), task="multiclass", num_classes=10)

        mlflow.log_metric("test_accuracy", test_acc)

        return {"test_acc": test_acc}

    def test_epoch_end(self, test_step_outputs):

        avg_test_acc = torch.stack([x["test_acc"] for x in test_step_outputs]).mean()
        mlflow.log_metric("average_test_accuracy", avg_test_acc)
        

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()

        log = {'avg_val_loss': val_loss}
        mlflow.log_metric("average_validation_loss", val_loss)
        return {'log ': log}

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.mnist_train,
            persistent_workers=True,
            num_workers=4,
            batch_size=self.kwargs.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.mnist_val,
            persistent_workers=True,
            batch_size=self.kwargs.batch_size,
            num_workers=4)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.mnist_test,
            persistent_workers=True,
            num_workers=4,
            batch_size=self.kwargs.batch_size),
        return test_loader
