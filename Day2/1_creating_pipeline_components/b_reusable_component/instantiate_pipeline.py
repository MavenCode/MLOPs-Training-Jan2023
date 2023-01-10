import kfp
from kfp import dsl

def preprocess_op():
    return dsl.ContainerOp(
        name = 'Preprocess Data',
        # docker image
        image = 'mavencodev/preprocess-component:v.0.2',
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
    output_component_file="preprocess-reusable.yaml" ,
    packages_to_install = ["pandas==0.23.4", "scikit-learn==0.22"])
                                        

# view the output YAML file and cp it into `c_load_component_from_yaml_folder`
