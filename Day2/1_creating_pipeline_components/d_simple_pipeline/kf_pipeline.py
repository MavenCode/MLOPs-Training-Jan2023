import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath
from kfp.components import create_component_from_func

from kf_components import download_dataset, process_data, train_model, evaluate_model, perform_sample_prediction

# create pipeline component ops
download_dataset_op = create_component_from_func(
    download_dataset, 
    base_image='python:3.10.8',
    packages_to_install=['pandas']
)

process_data_op = create_component_from_func(
    process_data, 
    base_image='python:3.10.8',
    packages_to_install=['pandas', 'scikit-learn']
)

train_model_op = create_component_from_func(
    train_model, 
    base_image='python:3.10.8',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)

evaluate_model_op = create_component_from_func(
    evaluate_model, 
    base_image='python:3.10.8',
    packages_to_install=['pandas', 'joblib', 'scikit-learn']
)

perform_sample_prediction_op = create_component_from_func(
    perform_sample_prediction, 
    base_image='python:3.10.8',
    packages_to_install=['joblib', 'scikit-learn']
)


# create pipeline kubeflow dsl
@dsl.pipeline(
    name='Basic pipeline',
    description='Basic pipeline'
)
def basic_pipeline():
    DOWNLOAD_DATASET = download_dataset_op()
    PROCESS_DATA = process_data_op(DOWNLOAD_DATASET.output)
    TRAIN_MODEL = train_model_op(PROCESS_DATA.outputs['df_training_data'])
    EVALUATE_MODEL = evaluate_model_op(TRAIN_MODEL.outputs['model'], PROCESS_DATA.outputs['df_test_data'])
    PERFORM_SAMPLE_PREDICTION = perform_sample_prediction_op(TRAIN_MODEL.outputs['model'])
    PERFORM_SAMPLE_PREDICTION.after(EVALUATE_MODEL)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(basic_pipeline, 'kfp_pipeline.yaml')
