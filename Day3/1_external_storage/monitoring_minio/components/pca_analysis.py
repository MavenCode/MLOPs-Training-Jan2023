def PCA(clean_data_dir: str) -> str:
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'sklean'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'joblib'])
    import pandas as pd
    import numpy as np
    import os
    import pickle
    import joblib
    import logging
    from sklearn.decomposition import PCA
    
    with open(os.path.join(clean_data_dir,'clean_data.pickle'), 'rb') as f:
        data = pickle.load(f)
        
    logging.info(data)
    
    
    X_train = data['X_train']
    X_test = data['X_test']  
    logging.info(len(X_train),len(X_test))
    pca = PCA(n_components=2, svd_solver= 'full')
    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index

    X_test_PCA = pca.transform(X_test)
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index
    
    def is_pos_def(A):

        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False
    
    def cov_matrix(data, verbose=False):
        covariance_matrix = np.cov(data, rowvar=False)
        
        if is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if is_pos_def(inv_covariance_matrix):
                return covariance_matrix, inv_covariance_matrix
            else:
                logging.info("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            logging.info("Error: Covariance Matrix is not positive definite!")
    def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
        inv_covariance_matrix = inv_cov_matrix
        vars_mean = mean_distr
        diff = data - vars_mean
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
        return md

    def MD_threshold(dist, extreme=False, verbose=False):
        k = 3. if extreme else 2.
        threshold = np.mean(dist) * k
        return threshold

    def is_pos_def(A):

        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False
    data_train = np.array(X_train_PCA.values)
    data_test = np.array(X_test_PCA.values)
    cov_matrix, inv_cov_matrix  = cov_matrix(data_train)
    mean_distr = data_train.mean(axis=0)
    dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
    dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    threshold = MD_threshold(dist_train, extreme = True)

    anomaly_train = pd.DataFrame()
    anomaly_train['Mob dist']= dist_train
    anomaly_train['Thresh'] = threshold
    # If Mob dist above threshold: Flag as anomaly
    anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
    anomaly_train.index = X_train_PCA.index

    anomaly = pd.DataFrame()
    anomaly['Mob dist']= dist_test
    anomaly['Thresh'] = threshold
    # If Mob dist above threshold: Flag as anomaly
    anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
    anomaly.index = X_test_PCA.index


    anomaly_alldata = pd.concat([anomaly_train, anomaly])
    data = anomaly_alldata[anomaly_alldata['Anomaly']==True]

    pca_dir = "/pvc_data/pca"
    
    os.makedirs(pca_dir, exist_ok=True)
    
    with open(os.path.join(pca_dir,'pca_metrics.pickle'), 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"pca_metrics.pickle {pca_dir}")
    
    logging.info(os.listdir(pca_dir))
    return pca_dir