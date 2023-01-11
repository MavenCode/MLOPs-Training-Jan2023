from typing import NamedTuple

import urllib.request
import tarfile
import os
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Data Download Logger__")
logger.info("Data Download Component log information...")

def download_dataset(minio_server: str, data_dir: str):
    """Download the data set to the KFP volume to share it among all steps"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = f'http://{minio_server}/monitoring/datasets.tar.gz'
    logger.info(url)
    stream = urllib.request.urlopen(url)
    logger.info('done downloading')
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    tar.extractall(path=data_dir)
    logger.info('done extracting')
    
    subprocess.call(["ls", "-dlha", data_dir])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Data Downloader Component"
    )
    parser.add_argument(
        "--minio_server",
        type=str,
        help="minio server endpoint",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="directory to download data to",
        required=True,
    )
    args = parser.parse_args()
    download_dataset(args.minio_server, args.data_dir)