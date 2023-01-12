import kfp
from kfp import dsl
from kfp import components

from components import mlflow_experiment
from components import data_prep
from components import test
from components import train

# Compose Pipeline Definition with All the defined Task Modules
def train_model_pipeline(
    pvc,
    mlflow_server,
    minio_server: str,
    minio_access_key: str,
    minio_secret_key: str,
    mlflow_experiment_name: str
):
    # For GPU support, please add the "-gpu" suffix to the base image
    BASE_IMAGE = "mavencodev/minio:v.0.1"

    mlflowOp = components.func_to_container_op(
        mlflow_experiment, base_image=BASE_IMAGE, 
        packages_to_install=["mlflow"]
    )(mlflow_server, mlflow_experiment_name)
    
    dataprepOP = components.func_to_container_op(
        data_prep, base_image=BASE_IMAGE, 
        packages_to_install=["torch", "torchvision", "mlflow"]
        )(mlflowOp.output,
        mlflow_server, minio_server, minio_access_key, minio_secret_key
        ).add_pvolumes({"/pvc_data": pvc})
    
    trainOP = components.func_to_container_op(
        train, base_image=BASE_IMAGE,
        packages_to_install=["torch", "pytorch-lightning", "mlflow", "torchvision"]
        )( 
        dataprepOP.output, mlflowOp.output,
        mlflow_server, minio_server, minio_access_key, minio_secret_key
        ).add_pvolumes({"/pvc_data": pvc})
    
    testOP = components.func_to_container_op(
        test, base_image=BASE_IMAGE,
        packages_to_install=["torch", "pytorch-lightning", "mlflow", "torchvision"]
        )(
        dataprepOP.output, trainOP.output, mlflowOp.output,
        mlflow_server, minio_server, minio_access_key, minio_secret_key
        ).add_pvolumes({"/pvc_data": pvc})
    

def op_transformer(op):
    op.add_pod_annotation(name="sidecar.istio.io/inject", value="false")
    return op


@dsl.pipeline(
    name="Mnist Pipeline",
    description="A sample pipeline to demonstrate mlflow logging in a kubeflow pipeline",
)
def mnist_pipeline(
    minio_access_key: str,
    minio_secret_key: str,
    experiment_name: str = "mlflow demo pipeline", 
):
    MLFLOW_SERVER="http://mlflow.cloudtraining-mavencode.com:5000"
    MINIO_SERVER="http://minio.cloudtraining-mavencode.com:9000"

    pvc_op = dsl.VolumeOp(name='pvc', 
            resource_name='datavolume',
            size='1Gi',
            modes=dsl.VOLUME_MODE_RWO)

    train_model_pipeline(
        pvc=pvc_op.volume,
        mlflow_server=MLFLOW_SERVER,
        minio_server=MINIO_SERVER,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        mlflow_experiment_name=experiment_name
    )
    
    dsl.get_pipeline_conf().add_op_transformer(op_transformer)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(mnist_pipeline, "mnist_pipeline_1.yaml")