def results(metrics_dir: str, pca_dir: str) -> None:
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    import pickle
    import os
    import pandas as pd
    import logging
    
    with open(os.path.join(metrics_dir,'metrics.pickle'), 'rb') as f:
        data = pickle.load(f)
        
    logging.info(data)
    logging.info("Autoencoder")
    
    if len(data) > 0:
        logging.info(f"There are anomalies in the data, {len(data)} \n\n")
        logging.info(data.head(20))
    else:
        
        logging.info(f"There are no anomalies")
        logging.info("\n\n **************** \n\n")
        
    with open(os.path.join(pca_dir,'pca_metrics.pickle'), 'rb') as f:
        data1 = pickle.load(f)
        
    logging.info(data1) 
    logging.info("PCA")
    
    if len(data1) > 0:
        logging.info(f"There are anomalies in the data, {len(data1)} \n\n")
        logging.info(data1.head(20))
    else:
        logging.info(f"There are no anomalies")