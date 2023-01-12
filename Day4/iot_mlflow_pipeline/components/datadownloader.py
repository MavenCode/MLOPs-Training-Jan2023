def download_dataset(exp_id: str, minio_server, minio_access_key, minio_secret_key) -> str:
    """Download the data set to the KFP volume to share it among all steps"""


    import urllib.request
    import tarfile
    import os
    import subprocess
    import logging
    import sys
    from datetime import datetime
    import pickle

    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlflow'])

    import mlflow

    mlflow.set_tracking_uri("http://mlflow.cloudtraining-mavencode.com:5000")
    
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_server

    data_dir = "pvc_data/ingest"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = f'{minio_server}/monitoring/datasets.tar.gz'
    logging.info(url)
    print(url)
    stream = urllib.request.urlopen(url)
    logging.info('done downloading')
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    tar.extractall(path=data_dir)
    logging.info('done extracting')
    print("done_extracting")
    
    subprocess.call(["ls", "-dlha", data_dir])
    
    run_name = f"Download_component_{datetime.now()}"

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        mlflow.log_param("dataset_url", url)
        mlflow.log_artifact(data_dir)
        
    return data_dir