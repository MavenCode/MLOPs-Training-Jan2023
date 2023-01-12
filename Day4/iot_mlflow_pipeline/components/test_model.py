def test(clean_data_dir: str, model_dir: str, exp_id: str, minio_server, minio_access_key, minio_secret_key) -> str:
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'keras'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'joblib'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlflow'])
    import numpy as np
    import pandas as pd
    from tensorflow import keras
    from keras.models import load_model
    from datetime import datetime
    import os
    import joblib
    import pickle
    import logging
    import mlflow
    
    mlflow.set_tracking_uri("http://mlflow.cloudtraining-mavencode.com:5000")
    
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_server
    

    logging.info(clean_data_dir)
    with open(os.path.join(clean_data_dir,'clean_data.pickle'), 'rb') as f:
        data = pickle.load(f)
        
    logging.info(data)
    
    model = keras.models.load_model(model_dir + "/model.h5")      
    logging.info(model)
    
    
    X_train = data['X_train']
    X_test = data['X_test']

    X_pred_train = model.predict(np.array(X_train))
    X_pred_train = pd.DataFrame(X_pred_train, 
                        columns=X_train.columns)
    X_pred_train.index = X_train.index

    scored_train = pd.DataFrame(index=X_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
    thress = np.max(scored_train['Loss_mae'])
    scored_train['Threshold'] = thress
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    X_pred = model.predict(np.array(X_test))
    X_pred = pd.DataFrame(X_pred, 
                        columns=X_test.columns)
    X_pred.index = X_test.index

    scored = pd.DataFrame(index=X_test.index)
    scored = scored[:1000]
    loss_mae = np.mean(np.abs(X_pred-X_test), axis = 1)
    scored['Loss_mae'] = loss_mae
    scored['Threshold'] = thress
    anomaly = scored['Loss_mae'] > scored['Threshold']
    scored['Anomaly'] = anomaly

    scored = pd.concat([scored_train, scored])
    data = scored[scored['Anomaly']==True] 

    metrics_dir = "pvc_data/tests"

    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(os.path.join(metrics_dir,'metrics.pickle'), 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"metrics.pickle {metrics_dir}")
    
    logging.info(os.listdir(metrics_dir))

    run_name = f"Test_component_{datetime.now()}"    
    
    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        mlflow.log_artifact(os.path.join(metrics_dir,'metrics.pickle'))

    return metrics_dir