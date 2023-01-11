from typing import NamedTuple
import os
import boto3
from botocore.client import Config
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__Model Export Logger__")
logger.info("Model Export Component log information...")

def export_model(
    model_dir: str,
    metrics_path: str,
    export_bucket: str,
    model_name: str,
    model_version: int,
    minio_server: str,
    minio_access_key: str,
    minio_secret_key: str,
    ):

    s3 = boto3.client(
        "s3",
        endpoint_url=f'http://{minio_server}',
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        config=Config(signature_version="s3v4"),
    )

    # Create export bucket if it does not yet exist
    response = s3.list_buckets()
    export_bucket_exists = False

    for bucket in response["Buckets"]:
        if bucket["Name"] == export_bucket:
            export_bucket_exists = True

    if not export_bucket_exists:
        s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)

    # Save model files to S3
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.relpath(local_path, model_dir)

            s3.upload_file(
                local_path,
                export_bucket,
                f"{model_name}/{model_version}/{s3_path}",
                ExtraArgs={"ACL": "public-read"},
            )

    response = s3.list_objects(Bucket=export_bucket)
    logger.info(f"All objects in {export_bucket}:")
    for file in response["Contents"]:
        logger.info("{}/{}".format(export_bucket, file["Key"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Model Export Component"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="directory to export the model to",
        required=True,
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        help="metrics path",
        required=True,
    )
    parser.add_argument(
        "--export_bucket",
        type=str,
        help="export bucket",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="name of model",
        required=True,
    )
    parser.add_argument(
        "--model_version",
        type=int,
        help="model version",
        required=True,
    )
    parser.add_argument(
        "--minio_server",
        type=str,
        help="minio server endpoint",
        required=True,
    )
    parser.add_argument(
        "--minio_access_key",
        type=str,
        help="access key for minio",
        required=True,
    )
    parser.add_argument(
        "--minio_secret_key",
        type=str,
        help="secret key for minio access",
        required=True,
    )
    args = parser.parse_args()
    
    export_model(
    args.model_dir,
    args.metrics_path,
    args.export_bucket,
    args.model_name,
    args.model_version,
    args.minio_server,
    args.minio_access_key,
    args.minio_secret_key,
    )