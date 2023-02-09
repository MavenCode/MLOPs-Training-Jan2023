import kfp
from kfp import dsl

CONTAINER_REGISTRY="TODO_insert_a_registry_name_here"
CONTAINER_TAG="charles-v1.0.0"

def ml_operator_op():
    return dsl.ContainerOp(
        name='Pytorch Operator',
        image=f'{CONTAINER_REGISTRY}/torch_job:latest',
        arguments=[],
        command=["python3", "pytorch_job.py"]
    )

@dsl.pipeline(
   name='Pytorch Operator ML Pipeline',
   description='A sample pipeline for distributed training on pytorch model with PyTorch Operator'
)
def pytorch_job_pipeline():

    training_task = ml_operator_op()

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pytorch_job_pipeline, 'pytorch_operator_pipeline.yaml')
    # client = kfp.Client(host='pipelines-api.kubeflow.svc.cluster.local:8888')
