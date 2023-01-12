import kfp
from kfp import dsl
from kfp import components

from components.mlflow_init import create_experiment
from components.datadownloader import download_dataset
from components.preprocess_dataset import preprocessing
from components.train import train
from components.pca_analysis import PCA
from components.test_model import test
from components.print_results import results

# Compose Pipeline Definition with All the defined Task Modules
def train_model_pipeline(
    pvc,
    minio_server: str,
    minio_access_key: str,
    minio_secret_key: str,
    experiment_name: str
):
    # For GPU support, please add the "-gpu" suffix to the base image
    BASE_IMAGE = "mavencodev/minio:v.0.1"

    mlflowOp = components.func_to_container_op(create_experiment, base_image=BASE_IMAGE)(experiment_name)
    downloadOp = components.func_to_container_op(download_dataset, base_image=BASE_IMAGE)(
        mlflowOp.output, 
        minio_server, minio_access_key, minio_secret_key).add_pvolumes({"/pvc_data": pvc})
    preprocessOp = components.func_to_container_op(preprocessing, base_image=BASE_IMAGE)(
        downloadOp.output, mlflowOp.output,
        minio_server, minio_access_key, minio_secret_key).add_pvolumes({"/pvc_data": pvc})
    trainOp = components.func_to_container_op(train, base_image=BASE_IMAGE)(
        preprocessOp.output, mlflowOp.output,
        minio_server, minio_access_key, minio_secret_key).add_pvolumes({"/pvc_data": pvc})
    pcaOp = components.func_to_container_op(PCA, base_image=BASE_IMAGE)(
        preprocessOp.output, mlflowOp.output,
        minio_server, minio_access_key, minio_secret_key).add_pvolumes({"/pvc_data": pvc})
    testOp = components.func_to_container_op(test, base_image=BASE_IMAGE)(
        preprocessOp.output, trainOp.output, mlflowOp.output,
        minio_server, minio_access_key, minio_secret_key).add_pvolumes({"/pvc_data": pvc})
    resultOp = components.func_to_container_op(results, base_image=BASE_IMAGE)(
        testOp.output, pcaOp.output, mlflowOp.output, 
        minio_server, minio_access_key, minio_secret_key
        ).add_pvolumes({"/pvc_data": pvc})


def op_transformer(op):
    op.add_pod_annotation(name="sidecar.istio.io/inject", value="false")
    return op


@dsl.pipeline(
    name="mlflow logging Pipeline",
    description="A sample pipeline to demonstrate mlflow logging in a kubeflow pipeline",
)
def monitoring_pipeline(
    minio_access_key: str,
    minio_secret_key: str,
    experiment_name: str = "iot pipeline experiment"
):
    MINIO_SERVER='http://minio.cloudtraining-mavencode.com:9000'

    pvc_op = dsl.VolumeOp(name='pvc', 
            resource_name='datavolume',
            size='1Gi',
            modes=dsl.VOLUME_MODE_RWO)

    train_model_pipeline(
        pvc=pvc_op.volume,
        minio_server=MINIO_SERVER,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        experiment_name=experiment_name
    )
    
    dsl.get_pipeline_conf().add_op_transformer(op_transformer)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(monitoring_pipeline, "iotpipeline.yaml")