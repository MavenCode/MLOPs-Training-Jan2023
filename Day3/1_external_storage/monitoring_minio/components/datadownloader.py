def download_dataset(minio_server: str) -> str:
    """Download the data set to the KFP volume to share it among all steps"""
    import urllib.request
    import tarfile
    import os
    import subprocess
    import logging

    data_dir = "pvc_data/ingest"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = f'{minio_server}/monitoring/datasets.tar.gz'
    logging.info(url)
    stream = urllib.request.urlopen(url)
    logging.info('done downloading')
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    tar.extractall(path=data_dir)
    logging.info('done extracting')
    
    
    subprocess.call(["ls", "-dlha", data_dir])
    
    
    return data_dir