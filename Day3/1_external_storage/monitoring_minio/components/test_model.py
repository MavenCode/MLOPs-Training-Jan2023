def test(clean_data_dir: str, model_dir: str) -> str:
    
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'keras'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'joblib'])
    import numpy as np
    import pandas as pd
    from tensorflow import keras
    from keras.models import load_model
    import os
    import joblib
    import pickle
    import logging
    
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
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
    scored['Threshold'] = thress
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    scored = pd.concat([scored_train, scored])
    data = scored[scored['Anomaly']==True] 

    metrics_dir = "pvc_data/tests"

    os.makedirs(metrics_dir, exist_ok=True)
    logging.info("open")
    with open(os.path.join(metrics_dir,'metrics.pickle'), 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"metrics.pickle {metrics_dir}")
    
    logging.info(os.listdir(metrics_dir))

    return metrics_dir