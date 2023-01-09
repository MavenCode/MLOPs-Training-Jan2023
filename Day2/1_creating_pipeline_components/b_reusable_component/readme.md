### Reusable Components Example

#### Creating a simple reusable component for data preprocessing.

Similar to the lightweight component, this example features creating a kubeflow pipeline component for data processing. 

In this  example, the component is containerised and pushed to a container registry, the container can then be used as a component in a kubeflow pipeline.

this example uses docker.

reusable components are slower to build when compared to lightweight components, but unlike lightweight components, they can be shared and used multiple times. 