def data_prep(
        exp_id: str,
        mlflow_server: str,
        minio_server: str,
        minio_access_key: str,
        minio_secret_key: str
) -> str:
    from torchvision import datasets, transforms
    import torch
    import os
    import pickle
    import mlflow
    from datetime import datetime

    mlflow.set_tracking_uri(mlflow_server)

    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_server

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_train, mnist_val = torch.utils.data.random_split(train_dataset, [55000,5000])

    mnist_test = datasets.MNIST(os.getcwd(), train=False, download=True, transform=transform)

    data_dir = "pvc_data/data"

    os.makedirs(f"{data_dir}/train")
    os.makedirs(f"{data_dir}/test")

    with open(f"{data_dir}/train/train.pkl", "wb") as f:
        pickle.dump(mnist_train, f)
    
    with open(f"{data_dir}/train/val.pkl", "wb") as f:
        pickle.dump(mnist_val, f)

    with open(f"{data_dir}/test/test.pkl", "wb") as f:
        pickle.dump(mnist_test, f)


    # Create an MLFlow run using the mlflow client
    run_name = f"Preprocess_component_{datetime.now()}"    


    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        mlflow.log_param("Dataset", "Mnist Classification Dataset")
        mlflow.log_artifact(os.path.join(data_dir, "train/train.pkl"), "data")
        mlflow.log_artifact(os.path.join(data_dir, "train/val.pkl"), "data")
        mlflow.log_artifact(os.path.join(data_dir, "test/test.pkl"), "data")
    


    return data_dir

