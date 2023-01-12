def train(
    data_dir: str,
    exp_id: str,
    mlflow_server: str,
    minio_server: str,
    minio_access_key: str,
    minio_secret_key: str
) -> str:
    import os
    import torch
    from torch.nn import functional as F
    from torch import nn
    import pickle
    import pytorch_lightning as pl
    import mlflow
    from datetime import datetime
    
    mlflow.set_tracking_uri(mlflow_server)

    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_server

    class MnistModel(pl.LightningModule):
        def __init__(self, batch_size, lr, mnist_train, mnist_val):
            super(MnistModel, self).__init__()

            self.lr = lr
            self.batch_size = batch_size
            self.mnist_train = mnist_train
            self.mnist_val = mnist_val

            self.conv_layer1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv_layer2 = nn.Conv2d(10, 20,kernel_size=5)
            self.conv_layer2_drop = nn.Dropout2d()

            self.layer_3 = nn.Linear(320,50)
            self.layer_4 = nn.Linear(50,10)
    
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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

        def validation_epoch_end(self, validation_step_outputs):
            val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()

            log = {'avg_val_loss': val_loss}

            mlflow.log_metric("average_validation_loss", val_loss)       
            return {'log ': log}

        def train_dataloader(self):
            train_loader = torch.utils.data.DataLoader(
                self.mnist_train,
                num_workers = 1,
                persistent_workers=True,
                batch_size=self.batch_size, shuffle=True)
            return train_loader

        def val_dataloader(self):
            val_loader = torch.utils.data.DataLoader(
                self.mnist_val,
                num_workers = 1,
                persistent_workers=True,
                batch_size=self.batch_size)
            return val_loader

    with open(os.path.join(data_dir, "train/train.pkl"), "rb") as f:
        mnist_train = pickle.load(f)

    with open(os.path.join(data_dir, "train/val.pkl"), "rb") as f:
        mnist_val = pickle.load(f)

    # Create an MLFlow run using the mlflow client
    run_name = f"Train_component_{datetime.now()}"    

    model = MnistModel(32, 0.001, mnist_train, mnist_val)

    trainer = pl.Trainer(max_epochs=1, num_nodes=1, devices=1)

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        mlflow.pytorch.autolog(log_models=True)
        trainer.fit(model)
    
    model_dir = "pvc_data/model"
    os.mkdir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "state_dict.pt"))

    return model_dir
