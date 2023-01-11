from typing import NamedTuple
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
import os
import joblib
import pickle
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Model Testing Logger__")
logger.info("Model Testing Component log information...")

def test(clean_data_dir: str, model_dir: str, metrics_path: str):
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'keras'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'joblib'])
    
    logger.info(clean_data_dir)
    with open(os.path.join(clean_data_dir,'clean_data.pickle'), 'rb') as f:
        data = pickle.load(f)
        
    logger.info(data)
    
    model = keras.models.load_model(model_dir + "/model.h5")      
    logger.info(model)
    
    
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
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
    scored['Threshold'] = thress
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    scored = pd.concat([scored_train, scored])
    data = scored[scored['Anomaly']==True] 
    
    logger.info(data)
    
    os.makedirs(metrics_path, exist_ok=True)
    
    with open(os.path.join(metrics_path,'metrics.pickle'), 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"metrics.pickle {metrics_path}")
    
    logger.info(os.listdir(metrics_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Model Testing Component"
    )
    parser.add_argument(
        "--clean_data_dir",
        type=str,
        help="clean data directory",
        required=True,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="model directory",
        required=True,
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        help="metrics path",
        required=True,
    )
    args = parser.parse_args()

    test(args.clean_data_dir, args.model_dir, args.metrics_path)