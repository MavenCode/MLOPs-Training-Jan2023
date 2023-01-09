import kfp
import kfp.dsl as dsl

# define python functions that represent pipeline task

@dsl.python_component
def dataloader(data: str) -> int:
    import pandas as pd
    rows = pd.read_csv(data)
    return rows

@dsl.python_component
def tiny_mode_trainer(data: str):
    pass

@dsl.python_component
def large_model_trainer(data: str):
    pass

# define a pipeline function that uses the task
@dsl.pipeline(
    name="Xperi Demo Pipeline",
    description="A simple ML Pipeline"
)
def xperi_ml_pipeline(
    data: str
):
    # use the pipeline parameter as an input to the task
    size_count = dataloader(data)

    # use a condition to branch the pipeline execution
    with dsl.Condition(size_count <= 100):
        w = tiny_mode_trainer(data)

    with dsl.Condition(size_count > 100):
        # use another pipeline parameter as an input to the task
        w = large_model_trainer(data)

if __name__ == "__main__":
    # compile and save the pipeline to a YAML file
    kfp.compiler.Compiler().compile(xperi_ml_pipeline, 'xperi_ml_pipeline.yaml')
    

