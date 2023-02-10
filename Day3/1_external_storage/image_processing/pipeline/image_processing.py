import kfp
from kfp import dsl

CONTAINER_REGISTRY=""
MINIO_SERVER='http://minio.cloudtraining-mavencode.com:9000'
MINIO_ACCESS_KEY='some_access_key'
MINIO_SECRET_KEY='some_secret_key'

def datadownload_op(minio_server, data_dir):
    return dsl.ContainerOp(
        name='Data Download',
        image=f'{CONTAINER_REGISTRY}/datadownloader:latest',
        arguments=[
            '--minio_server', minio_server,
            '--data_dir', data_dir
        ],
        command=["python3", "datadownloader.py"]
    )

def data_preprocess_op(data_dir, batch_size, clean_data_dir):
    return dsl.ContainerOp(
        name='Data Preprocessing',
        image=f'{CONTAINER_REGISTRY}/data_prep:latest',
        arguments=[
            '--data_dir', data_dir,
            '--clean_data_dir', clean_data_dir,
            '--batch_size', batch_size
        ],
        command=["python3", "preprocess_dataset.py"]
    )

def train_op(clean_data_dir):
    return dsl.ContainerOp(
        name='Model Training',
        image=f'{CONTAINER_REGISTRY}/model_train:latest',
        arguments=[
            '--clean_data_dir', clean_data_dir
        ],
        command=["python3", "model-train.py"]
    )

def op_transformer(op):
    op.add_pod_annotation(name="sidecar.istio.io/inject", value="false")
    return op

@dsl.pipeline(
   name='End-to-End image_processing pipeline',
   description='A sample pipeline to demonstrate multi-class image classification'
)
def image_processing_pipeline(
    batch_size: int = 64,
    data_dir: str = "/train",
    clean_data_dir: str = "/train/data",
    minio_server: str = MINIO_SERVER,
    minio_access_key: str = MINIO_ACCESS_KEY,
    minio_secret_key: str = MINIO_SECRET_KEY
):

    pvc_op = dsl.VolumeOp(name='Persistent Storage', 
            resource_name='data-volume',
            size='2Gi',
            modes=dsl.VOLUME_MODE_RWO)

    datadownload_task    = datadownload_op(minio_server=minio_server, data_dir=data_dir).add_pvolumes({"/train": pvc_op.volume})

    data_preprocess_task = data_preprocess_op(data_dir=f'{data_dir}/data', clean_data_dir=clean_data_dir, batch_size=batch_size).add_pvolumes({"/train": pvc_op.volume}).after(datadownload_task)

    train_task           = train_op(clean_data_dir=clean_data_dir).add_pvolumes({"/train": pvc_op.volume}).after(data_preprocess_task)

    dsl.get_pipeline_conf().add_op_transformer(op_transformer)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(image_processing_pipeline, 'image_processing_pipeline.yaml')
    # client = kfp.Client(host='pipelines-api.kubeflow.svc.cluster.local:8888')