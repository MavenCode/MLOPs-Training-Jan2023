import pickle
import os
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Results Logger__")
logger.info("Results Component log information...")

def results(pca_dir: str, metrics_path: str):
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    
    with open(os.path.join(metrics_path,'metrics.pickle'), 'rb') as f:
        data = pickle.load(f)
        
    logger.info(data)
    logger.info("Autoencoder")
    
    if len(data) > 0:
        logger.info(f"There are anomalies in the data, {len(data)} \n\n")
        logger.info(data.head(20))
    else:
        
        logger.info(f"There are no anomalies")
        logger.info("\n\n **************** \n\n")
        
    with open(os.path.join(pca_dir,'pca_metrics.pickle'), 'rb') as f:
        data1 = pickle.load(f)
        
    logger.info(data1) 
    logger.info("PCA")
    
    if len(data1) > 0:
        logger.info(f"There are anomalies in the data, {len(data1)} \n\n")
        logger.info(data1.head(20))
    else:
        logger.info(f"There are no anomalies")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Results logger.infoing Component"
    )
    parser.add_argument(
        "--pca_dir",
        type=str,
        help="PCA directory",
        required=True,
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        help="metrics path",
        required=True,
    )
    args = parser.parse_args()

    results(args.pca_dir, args.metrics_path)