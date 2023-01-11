import kfp
from kfp import dsl
import yaml, json

# load the experiment namespace yaml configuration as json
with open('../katib-experiment/namespace.yaml') as ns:
    experiment_namespace = json.dumps(yaml.load(ns))

# load the experiment resource yaml configuration as json
with open('../katib-experiment/pytorch_hyperparam.yaml') as res:
    experiment_resource = json.dumps(yaml.load(res))


@dsl.pipeline(
   name='Pytorch Katib Experiment',
   description='A sample pipeline for hyperparameter tuning with katib'
)
def experiment_pipeline():
    namespace_op = dsl.ResourceOp(
        name='configure namespace',
        k8s_resource=json.loads(experiment_namespace),
        action='apply'
    )

    experiment_op = dsl.ResourceOp(
        name='deploy katib experiment',
        k8s_resource=json.loads(experiment_resource),
        action='apply'
    ).after(namespace_op)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(experiment_pipeline, 'experiment_pipeline.yaml')