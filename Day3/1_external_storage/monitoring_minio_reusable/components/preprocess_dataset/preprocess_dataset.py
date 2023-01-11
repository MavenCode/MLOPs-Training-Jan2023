from typing import NamedTuple
import numpy as np
import pandas as pd
import os
import pickle
from sklearn import preprocessing
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Data Preprocessing Logger__")
logger.info("Data Preprocessing Component log information...")

def data_preprocessing(data_dir: str, clean_data_dir: str):
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'sklearn'])
    
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
    
    
    os.makedirs(clean_data_dir, exist_ok=True)

    with open(os.path.join(clean_data_dir,'clean_data.pickle'), 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"clean_data.pickle {clean_data_dir}")
    
    logger.info(os.listdir(clean_data_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Data Preprocessing Component"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data directory",
        required=True,
    )
    parser.add_argument(
        "--clean_data_dir",
        type=str,
        help="clean data directory",
        required=True,
    )
    args = parser.parse_args()

    data_preprocessing(args.data_dir, args.clean_data_dir)