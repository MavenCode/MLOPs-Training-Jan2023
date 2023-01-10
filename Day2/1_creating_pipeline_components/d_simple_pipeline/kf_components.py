import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath
from kfp.components import create_component_from_func

#1. ingest data for the pipeline component
def download_dataset(
    df_all_data_path: OutputPath(str)):
    
    import pandas as pd
    
    url="https://raw.githubusercontent.com/MavenCode/MLOPs-Training-Jan2023/main/data/hr/salary.csv"
    
    df_all_data = pd.read_csv(url)
    print(df_all_data)
    df_all_data.to_csv(df_all_data_path, header=True, index=False)

#2. prep process data
def process_data(
    df_all_data_path: InputPath(str), 
    df_training_data_path: OutputPath(str), 
    df_test_data_path: OutputPath(str)):
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df_all_data = pd.read_csv(df_all_data_path)
    print(df_all_data)
    
    X = df_all_data['management_experience_months'].values 
    y = df_all_data['monthly_salary'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    df_training_data = pd.DataFrame({ 'monthly_salary': y_train, 'management_experience_months': X_train})
    df_training_data.to_csv(df_training_data_path, header=True, index=False)
    df_test_data = pd.DataFrame({ 'monthly_salary': y_test, 'management_experience_months': X_test})
    df_test_data.to_csv(df_test_data_path, header=True, index=False)

#3. train model
def train_model(
    df_training_data_path: InputPath(str),
    model_path: OutputPath(str)):
    
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from joblib import dump
    
    df_training_data = pd.read_csv(df_training_data_path)
    print(df_training_data)
    
    X_train = df_training_data['management_experience_months'].values
    y_train = df_training_data['monthly_salary'].values
    
    model = LinearRegression().fit(X_train.reshape(-1, 1), y_train)
    print(model)
    dump(model, model_path)

#4. evaluate model
def evaluate_model(
    model_path: InputPath(str),
    df_test_data_path: InputPath(str)):
    
    import pandas as pd
    from joblib import load
    
    df_test_data = pd.read_csv(df_test_data_path)
    
    X_test = df_test_data['management_experience_months'].values
    y_test = df_test_data['monthly_salary'].values
    
    model = load(model_path)
    print(model.score(X_test.reshape(-1, 1), y_test))

#5. perform inference
def perform_sample_prediction(
    model_path: InputPath(str)):
    from joblib import load
    
    model = load(model_path)
    print(model.predict([[42]])[0])