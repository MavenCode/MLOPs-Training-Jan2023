import kfp
from kfp import dsl

def preprocess_op():
    from kfp import dsl
    return dsl.ContainerOp(
        name = 'Preprocess Data',
        # docker image
        image = 'public.ecr.aws/c8r6f6w7/preprocess:charles-latest',
        arguments = [],
        # component outputs
        file_outputs={
            'X_train': '/preprocess_data/X_train.npy',
            'X_test': '/preprocess_data/X_test.npy',
            'y_train': '/preprocess_data/y_train.npy',
            'y_test': '/preprocess_data/y_test.npy'     
        }
    )

# exporting the component as a yaml file
# TODO: fix this
if __name__ == "__main__":
    kfp.components.create_component_from_func(
    preprocess_op, #function name
    base_image="python:3.10",
    packages_to_install = ["kfp"],
    output_component_file="preprocess-reusable.yaml")