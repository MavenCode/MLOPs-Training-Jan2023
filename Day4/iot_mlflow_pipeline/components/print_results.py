def results(metrics_dir: str, pca_dir: str, 
            exp_id:str, minio_server, minio_access_key, minio_secret_key) -> None:
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlflow'])

    import pickle
    import os
    import pandas as pd
    import logging
    import mlflow
    from datetime import datetime

    mlflow.set_tracking_uri("http://mlflow.cloudtraining-mavencode.com:5000")
    
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_server

    with open(os.path.join(metrics_dir,'metrics.pickle'), 'rb') as f:
        data = pickle.load(f)
    
    with open(os.path.join(pca_dir,'pca_metrics.pickle'), 'rb') as f:
        data1 = pickle.load(f)
    

    run_name = f"Results_component_{datetime.now()}"    
    
    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        
        logging.info(data)
        logging.info("Autoencoder")
        
        logdata = pd.DataFrame(data.head(20))
        logdata.to_csv("./metrics_data.csv")

        logdata2 = pd.DataFrame(data1.head(20))
        logdata2.to_csv("./pca_data.csv")
        
        if len(data) > 0:
            logging.info(f"There are anomalies in the data, {len(data)} \n\n")
            logging.info(data.head(20))
            mlflow.log_param("metrics_data", "There are anomalies in the Metrics data")
            mlflow.log_artifact("./metrics_data.csv", "metrics_anomaly_data")
        else:
            logging.info(f"There are no anomalies")
            logging.info("\n\n **************** \n\n")
            mlflow.log_param("metrics_data", "There are no anomalies a in the Metrics data")
            mlflow.log_artifact("./metrics_data.csv", "metrics_data_sample")

        logging.info(data1) 
        logging.info("PCA")
        
        if len(data1) > 0:
            logging.info(f"There are anomalies in the data, {len(data1)} \n\n")
            logging.info(data1.head(20))
            mlflow.log_param("pca_data", "There are anomalies in the PCA data")
            mlflow.log_artifact("./pca_data.csv", "pca_anomaly_data")
        else:
            logging.info(f"There are no anomalies")
            mlflow.log_param("pca_data", "There are no anomalies in the PCA data")
            mlflow.log_artifact("./pca_data.csv", "pca_data_sample")