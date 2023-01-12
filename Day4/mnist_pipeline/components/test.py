
def test(
        data_dir: str, 
        model_dir: str,
        exp_id: str,
        mlflow_server: str,
        minio_server: str,
        minio_access_key: str,
        minio_secret_key: str
) -> None:
    import torch
    import pytorch_lightning as pl
    import pickle
    from torch import nn
    from torch.nn import functional as F
    from torchmetrics.functional import accuracy
    import os
    import mlflow
    from datetime import datetime

    mlflow.set_tracking_uri(mlflow_server)

    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_server

    class MnistModel(pl.LightningModule):
        def __init__(self, mnist_test, batch_size):
            super(MnistModel, self).__init__()
            self.mnist_test = mnist_test
            self.batch_size = batch_size
            self.conv_layer1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv_layer2 = nn.Conv2d(10, 20,kernel_size=5)
            self.conv_layer2_drop = nn.Dropout2d()

            self.layer_3 = nn.Linear(320,50)
            self.layer_4 = nn.Linear(50,10)
                                                    
        def test_dataloader(self):
            test_loader = torch.utils.data.DataLoader(
                self.mnist_test,
                num_workers = 1,
                persistent_workers=True,
                batch_size=self.batch_size),
            return test_loader

        def forward(self, x):
            output_conv1 = F.relu(F.max_pool2d(self.conv_layer1(x),2))
            output_conv2 = F.relu(F.max_pool2d(self.conv_layer2_drop(self.conv_layer2(output_conv1)),2))

            input_fc1 = output_conv2.view(-1,320)
            output_fc1 = F.relu(self.layer_3(input_fc1))
            output_dropout = F.dropout(output_fc1, training=self.training)
            output_layer = self.layer_4(output_dropout)
            return F.log_softmax(output_layer)


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
    
    
    with open(os.path.join(data_dir, "test/test.pkl"), "rb") as f:
        mnist_test = pickle.load(f)

    # Create an MLFlow run using the mlflow client
    run_name = f"Test_component_{datetime.now()}"    

    model = MnistModel(mnist_test, 32)
    model.load_state_dict(torch.load(os.path.join(model_dir, "state_dict.pt")))

    trainer = pl.Trainer(max_epochs=1, num_nodes=1, devices=1)

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        trainer.test(model)
