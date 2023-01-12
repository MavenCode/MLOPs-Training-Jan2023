def preprocessing(data_dir: str, exp_id: str, minio_server, minio_access_key, minio_secret_key) -> str:
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'sklearn'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlflow'])
    
    import numpy as np
    import pandas as pd
    import os
    import pickle
    from sklearn import preprocessing
    from datetime import datetime
    import logging
    import mlflow

    mlflow.set_tracking_uri("http://mlflow.cloudtraining-mavencode.com:5000")
    
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_server

    df = pd.read_csv(f"{data_dir}/datasets/data.csv")
    df = df.drop(['day', 'hour', 'sample_Number', 'month', 'timestamp','mode', 'pCut::Motor_Torque'
                 ,'pCut::CTRL_Position_controller::Lag_error','pSvolFilm::CTRL_Position_controller::Lag_error'], axis=1)
    df = df.fillna(0)
    train_percentage = 0.30
    train_size = int(len(df.index)*train_percentage)
    x_train = df[:train_size]
    x_test = df[train_size:490000]
    scaler = preprocessing.MinMaxScaler()

    X_train = pd.DataFrame(scaler.fit_transform(x_train), 
                                columns=x_train.columns, 
                                index=x_train.index)
    # Random shuffle training data
    X_train.sample(frac=1)

    X_test = pd.DataFrame(scaler.transform(x_test), 
                              columns=x_test.columns, 
                              index=x_test.index)
    data = {"X_train": X_train,"X_test": X_test}
    
    clean_data_dir = "/pvc_data/prep"
    os.makedirs(clean_data_dir, exist_ok=True)

    with open(os.path.join(clean_data_dir,'clean_data.pickle'), 'wb') as f:
        pickle.dump(data, f)

    run_name = f"Preprocess_component_{datetime.now()}"    

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        mlflow.log_artifact(os.path.join(clean_data_dir,'clean_data.pickle') , "clean_data")
        

    logging.info(f"clean_data.pickle {clean_data_dir}")
    
    logging.info(os.listdir(clean_data_dir))

    return clean_data_dir