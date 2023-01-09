### Lightweight Components Example

#### Creating a simple lightweight component for data preprocessing.

The preprocess function downloads the churn_modeling csv file from github, carry out preprocessing on the data and outputs the train data and test data as pickled files

the function is then converted to a lightweigt kubeflow pipeline component using kfp and compiled as a yaml specifcation that can then be deployed on kubeflow.