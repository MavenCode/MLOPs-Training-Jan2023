import kfp
from kfp import dsl
from kfp import components

from components.datadownloader import download_dataset
from components.preprocess_dataset import preprocessing
from components.train import train
from components.pca_analysis import PCA
from components.test_model import test
from components.print_results import results
from components.export_models import export_model

# Compose Pipeline Definition with All the defined Task Modules
def train_model_pipeline(
    pvc,
    export_bucket: str,
    model_name: str,
    model_version: int,
    minio_server: str,
    minio_access_key: str,
    minio_secret_key: str,
):
    # For GPU support, please add the "-gpu" suffix to the base image
    BASE_IMAGE = "mavencodev/minio:v.0.1"

    downloadOp = components.func_to_container_op( download_dataset, base_image=BASE_IMAGE)(minio_server).add_pvolumes({"/pvc_data": pvc})
    preprocessOp = components.func_to_container_op(preprocessing, base_image=BASE_IMAGE)(downloadOp.output).add_pvolumes({"/pvc_data": pvc})
    trainOp = components.func_to_container_op(train, base_image=BASE_IMAGE)(preprocessOp.output).add_pvolumes({"/pvc_data": pvc})
    pcaOp = components.func_to_container_op(PCA, base_image=BASE_IMAGE)(preprocessOp.output).add_pvolumes({"/pvc_data": pvc})
    testOp = components.func_to_container_op(test, base_image=BASE_IMAGE)(preprocessOp.output, trainOp.output).add_pvolumes({"/pvc_data": pvc})
    resultOp = components.func_to_container_op(results, base_image=BASE_IMAGE)(testOp.output, pcaOp.output).add_pvolumes({"/pvc_data": pvc})
    exportOp = components.func_to_container_op(export_model, base_image=BASE_IMAGE)(
        trainOp.output, testOp.output, export_bucket, model_name, model_version,
        minio_server, minio_access_key, minio_secret_key
    ).add_pvolumes({"/pvc_data": pvc})


def op_transformer(op):
    op.add_pod_annotation(name="sidecar.istio.io/inject", value="false")
    return op


@dsl.pipeline(
    name="End-to-End Industrial IoT Pipeline",
    description="A sample pipeline to demonstrate multi-step model training, evaluation and export",
)
def monitoring_pipeline(
    export_bucket: str = "monitoring",
    model_name: str = "monitoring",
    model_version: int = 1,
):
    MINIO_SERVER='minio.cloudtraining-mavencode.com:9000'
    MINIO_ACCESS_KEY=''
    MINIO_SECRET_KEY=''

    pvc_op = dsl.VolumeOp(name='Persistent Volume Claim', 
            resource_name='data-volume',
            size='1Gi',
            modes=dsl.VOLUME_MODE_RWO)

    train_model_pipeline(
        pvc=pvc_op.volume,
        export_bucket=export_bucket,
        model_name=model_name,
        model_version=model_version,
        minio_server=MINIO_SERVER,
        minio_access_key=MINIO_ACCESS_KEY,
        minio_secret_key=MINIO_SECRET_KEY,
    )
    
    dsl.get_pipeline_conf().add_op_transformer(op_transformer)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(monitoring_pipeline, "monitoringpipeline.yaml")