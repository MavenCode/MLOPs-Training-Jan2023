def train(clean_data_dir: str) -> str:
    #importing libraries
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'keras'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'joblib'])
    import pandas as pd
    import pickle
    import os
    import joblib
    import numpy as np
    import tensorflow as tf
    from keras.layers import Input, Dropout
    from keras.layers.core import Dense 
    from keras.models import Model, Sequential, load_model
    from keras import regularizers
    import logging

    with open(os.path.join(clean_data_dir,'clean_data.pickle'), 'rb') as f:
        data = pickle.load(f)
        
    logging.info(data) 
    
    np.random.seed(10)
    tf.random.set_seed(10)
    X_train = data['X_train']
    act_func = 'elu'

    # Input layer:
    model=Sequential()
    # First hidden layer, connected to input vector X. 
    model.add(Dense(10,activation=act_func,
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(0.0),
                  input_shape=(X_train.shape[1],)
                )
          )

    model.add(Dense(2,activation=act_func, kernel_initializer='glorot_uniform'))
    model.add(Dense(10,activation=act_func, kernel_initializer='glorot_uniform'))
    model.add(Dense(X_train.shape[1],  kernel_initializer='glorot_uniform'))

    model.compile(loss='mse',optimizer='adam')

    # Train model for 100 epochs, batch size of 10: 
    NUM_EPOCHS=5
    BATCH_SIZE=10

    model.fit(np.array(X_train),np.array(X_train),
                    batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS,
                    validation_split=0.1,
                    verbose = 1)

    model_dir = "pvc_data/models"

    os.makedirs(model_dir, exist_ok=True)

    model.save(model_dir + "/model.h5")
    logging.info(f"Model saved {model_dir}")
    logging.info(os.listdir(model_dir))

    return model_dir