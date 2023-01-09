import kfp
from kfp import dsl


# Build a lightweight component from a python function
# a.  Define the python function with all its dependencies installed and imported within it
# b.  Download your data
# c.  Write your function and return its output

def preprocess(data_path,train_data,test_data):
    import os
    import pickle
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler 

    #download the data
    data = pd.read_csv("https://raw.githubusercontent.com/MavenCode/MLOpsTraining-Dec2022/master/data/telco/churn_modeling.csv")

    #dropping some columns that are not needed
    data = data.drop(columns=['RowNumber','CustomerId','Surname'], axis=1)
    #data features
    X = data.iloc[:,:-1]
    #target data
    y = data.iloc[:,-1:]   
    #encoding the categorical columns
    le = LabelEncoder()
    ohe = OneHotEncoder()
    X['Gender'] = le.fit_transform(X['Gender'])
    geo_df = pd.DataFrame(ohe.fit_transform(X[['Geography']]).toarray())

    #getting feature name after onehotencoding
    geo_df.columns = ohe.get_feature_names_out(['Geography'])

    #merging geo_df with the main data
    X = X.join(geo_df) 
    #dropping the old columns after encoding
    X.drop(columns=['Geography'], axis=1, inplace=True)

    #splitting the data 
    X_train,X_test,y_train,y_test = train_test_split( X,y, test_size=0.2, random_state = 42)
    #feature scaling
    sc =StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #saving the values from the dataframe
    y_train = y_train.values
    y_test = y_test.values
    
    os.mkdir(data_path)
    #Save the train_data as a pickle file to be used by the train component.
    with open(f'{data_path}/{train_data}', 'wb') as f:
        pickle.dump((X_train,  y_train), f)
        
    #Save the test_data as a pickle file to be used by the predict component.
    with open(f'{data_path}/{test_data}', 'wb') as f:
        pickle.dump((X_test,  y_test), f)
    
    return(print('Done!'))


# instantiate and assign the component to `preprocess_op` variable
preprocess_op = kfp.components.create_component_from_func(preprocess, base_image="python:3.10.8", packages_to_install=["pandas", "scikit-learn"])

# create pipeline kubeflow dsl
@dsl.pipeline(
    name='Basic pipeline',
    description='Basic pipeline'
)
def basic_pipeline():
    PRE_PROCESS = preprocess_op(
        data_path="data", train_data="train_data", test_data="test_data"
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(basic_pipeline, 'kfp_pipeline.yaml')

