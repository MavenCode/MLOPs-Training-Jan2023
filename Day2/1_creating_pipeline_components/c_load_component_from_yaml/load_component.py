import kfp
from kfp import dsl

# instantiate and assign the component to `preprocess_op` variable
preprocess_op = kfp.components.load_component_from_file("preprocess-reusable.yaml")

# create pipeline kubeflow dsl
@dsl.pipeline(
    name='Basic pipeline',
    description='Basic pipeline'
)
def basic_pipeline():
    PRE_PROCESS = preprocess_op()

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(basic_pipeline, 'kfp_pipeline.yaml')