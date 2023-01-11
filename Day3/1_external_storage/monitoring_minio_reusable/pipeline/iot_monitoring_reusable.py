import kfp
from kfp import dsl

CONTAINER_REGISTRY="<Todo_Insert_container_registry_here>"
MINIO_SERVER='minio.cloudtraining-mavencode.com:9000'
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

def data_preprocess_op(data_dir, clean_data_dir):
    return dsl.ContainerOp(
        name='Data Preprocessing',
        image=f'{CONTAINER_REGISTRY}/preprocess_dataset:latest',
        arguments=[
            '--data_dir', data_dir,
            '--clean_data_dir', clean_data_dir
        ],
        command=["python3", "preprocess_dataset.py"]
    )

def pca_analysis_op(clean_data_dir, pca_dir):
    return dsl.ContainerOp(
        name='PCA Analysis',
        image=f'{CONTAINER_REGISTRY}/pca_analysis:latest',
        arguments=[
            '--clean_data_dir', clean_data_dir,
            '--pca_dir', pca_dir
        ],
        command=["python3", "pca_analysis.py"]
    )

def train_op(clean_data_dir, model_dir):
    return dsl.ContainerOp(
        name='Model Training',
        image=f'{CONTAINER_REGISTRY}/train_model:latest',
        arguments=[
            '--clean_data_dir', clean_data_dir,
            '--model_dir', model_dir
        ],
        command=["python3", "train.py"]
    )

def model_test_op(clean_data_dir, model_dir, metrics_path):
    return dsl.ContainerOp(
        name='Model Testing',
        image=f'{CONTAINER_REGISTRY}/test_model:latest',
        arguments=[
            '--clean_data_dir', clean_data_dir,
            '--model_dir', model_dir,
            '--metrics_path', metrics_path
        ],
        command=["python3", "test_model.py"]
    )


def model_export_op(model_dir, metrics_path, export_bucket, model_name, model_version, minio_server, 
                    minio_access_key, minio_secret_key):
    return dsl.ContainerOp(
        name='Model Export',
        image=f'{CONTAINER_REGISTRY}/export_models:latest',
        arguments=[
            '--model_dir', model_dir,
            '--metrics_path', metrics_path,
            '--export_bucket', export_bucket,
            '--model_name', model_name,
            '--model_version', model_version,
            '--minio_server', minio_server,
            '--minio_access_key', minio_access_key,
            '--minio_secret_key', minio_secret_key
        ],
        command=["python3", "export_models.py"]
    )

def print_results_op(pca_dir, metrics_path):
    return dsl.ContainerOp(
        name='Result Printing',
        image=f'{CONTAINER_REGISTRY}/print_results:latest',
        arguments=[
            '--pca_dir', pca_dir,
            '--metrics_path', metrics_path
        ],
        command=["python3", "print_results.py"]
    )

def op_transformer(op):
    op.add_pod_annotation(name="sidecar.istio.io/inject", value="false")
    return op



@dsl.pipeline(
   name='End-to-End Industrial IoT Pipeline',
   description='A sample pipeline to demonstrate multi-step model training, evaluation and export'
)
def monitoring_pipeline(
    model_dir: str = "/train/model",
    data_dir: str = "/train/data",
    clean_data_dir: str = "/train/data",
    metrics_path: str ="/train/metrics",
    pca_dir: str ="/train/metrics",
    export_bucket: str = "monitoring",
    model_name: str = "monitoring",
    model_version: int = 1,
    minio_server: str=MINIO_SERVER,
    minio_access_key: str=MINIO_ACCESS_KEY,
    minio_secret_key: str=MINIO_SECRET_KEY
):

    pvc_op = dsl.VolumeOp(name='Persistent Storage', 
            resource_name='data-volume',
            size='2Gi',
            modes=dsl.VOLUME_MODE_RWO)

    datadownload_task = datadownload_op(MINIO_SERVER, data_dir).add_pvolumes({"/train": pvc_op.volume})

    data_preprocess_task = data_preprocess_op(data_dir=data_dir, clean_data_dir=clean_data_dir).add_pvolumes({"/train": pvc_op.volume}).after(datadownload_task)

    pca_analysis_task = pca_analysis_op(clean_data_dir=clean_data_dir, pca_dir=pca_dir).add_pvolumes({"/train": pvc_op.volume}).after(data_preprocess_task)

    train_task = train_op(clean_data_dir=clean_data_dir, model_dir=model_dir).add_pvolumes({"/train": pvc_op.volume}).after(data_preprocess_task)

    test_task = model_test_op(clean_data_dir=clean_data_dir, model_dir=model_dir, metrics_path=metrics_path).add_pvolumes({"/train": pvc_op.volume}).after(train_task)

    model_export_task = model_export_op(
                            model_dir=model_dir, 
                            metrics_path=metrics_path, 
                            export_bucket=export_bucket, 
                            model_name=model_name, 
                            model_version=model_version, 
                            minio_server=minio_server, 
                            minio_access_key=minio_access_key, 
                            minio_secret_key=minio_secret_key).add_pvolumes({"/train": pvc_op.volume}).after(test_task)

    results_task = print_results_op(pca_dir=pca_dir, metrics_path=metrics_path).add_pvolumes({"/train": pvc_op.volume}).after(test_task, pca_analysis_task)

    dsl.get_pipeline_conf().add_op_transformer(op_transformer)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(monitoring_pipeline, 'monitoring_pipeline.yaml')
    # client = kfp.Client(host='pipelines-api.kubeflow.svc.cluster.local:8888')