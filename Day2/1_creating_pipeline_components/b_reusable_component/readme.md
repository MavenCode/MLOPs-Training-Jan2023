### Creating Reusable Components and Serializing it into YAML

#### This will us how to build reusuable components and create sharable output artifacts from it

1. Run the make commands (connect to the docker registry, build and skaffold push to docker registry
2. Once docker image is built and pushed from step 1 above
   - `cd generate_component_yaml` folder, run the python file to generate the reusable component spec
   - inspect the component YAML spec to make sure everything looks good
   - cp spec to the `load_up_component_pipeline` folder
   - `cd load_up_component_pipeline` run the pipeline code to generate the workflow from the pipeline
   - upload the generated workflow yaml file to kubeflow pipeline
   - Run the pipeline and check the logs to make sure the component code executed correctly