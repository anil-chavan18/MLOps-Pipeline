# Import necessary libraries for the MLOps pipeline
import kfp
from kfp import dsl
from kfp.components import create_component_from_func

# The base image is the same for all components to ensure a consistent environment
BASE_IMAGE = 'mlops-pipeline:latest'

# ==============================================================================
# Pipeline Components (Steps 1-6)
# ==============================================================================

# Step 2: Data Pre-processing
def data_preparation() -> str:
    """
    This component handles data ingestion, cleaning, and pre-processing.
    It takes a raw Excel file as input and outputs a cleaned CSV artifact.
    """
    import pandas as pd
    
    print("Starting data preparation...")
    # Step 1: Data Ingestion is handled by the pipeline passing this file to the container.
    # The 'pandas.read_excel' function automatically handles .xlsx and .xlsm files.
    df = pd.read_excel('/app/large_bank_data.xlsm')
    
    # Data cleaning
    df.dropna(inplace=True)
    
    # Step 3: Feature Engineering
    # We create a new, potentially useful feature from existing data.
    df['age_loan_ratio'] = df['age'] / (df['loan_status'] + 1e-6)
    
    # Save the prepared data to the output path provided by Kubeflow
    output_path = '/tmp/processed_data.csv'
    df.to_csv(output_path, index=False)
    print("Data preparation and feature engineering complete.")
    return output_path

# Step 4: Model Training & Step 6: Logging and Tracking
def model_training(data_path: str, mlflow_tracking_uri: str) -> str:
    """
    This component trains the model and logs the experiment to MLflow.
    It takes the pre-processed data and an MLflow URI as input.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import mlflow
    import mlflow.sklearn
    import joblib
    
    print("Starting model training...")
    # Step 6: Logging and Tracking - Set the connection to the MLflow server.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Term Deposit Prediction")
    
    # Start an MLflow run to log all aspects of this training session.
    with mlflow.start_run() as run:
        df = pd.read_csv(data_path)
        X = df[['age', 'balance', 'loan_status', 'campaign_calls']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Log model parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("train_split", 0.8)
        
        # Train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Log the trained model and save it as a local artifact
        mlflow.sklearn.log_model(model, "logistic-regression-model")
        model_path = '/tmp/model.pkl'
        joblib.dump(model, model_path)
        
        print(f"Model trained and logged to MLflow run: {run.info.run_id}")
        return model_path

# Step 5: Model Evaluation & Step 6: Logging and Tracking (Metrics)
def model_validation(model_path: str, data_path: str):
    """
    This component loads the model and evaluates its performance.
    It prints the accuracy and logs it as a metric to MLflow.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    import mlflow
    import os
    
    print("Starting model validation...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df[['age', 'balance', 'loan_status', 'campaign_calls']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Report the final accuracy.
    print(f"Model Accuracy: {accuracy:.4f}")
    if accuracy >= 0.85:
        print("Model meets the 85% accuracy threshold. 🎉")
    else:
        print("Model accuracy is below the 85% threshold. 😥")

# Define Kubeflow pipeline components from the Python functions
data_prep_op = create_component_from_func(data_preparation, base_image=BASE_IMAGE)
model_train_op = create_component_from_func(model_training, base_image=BASE_IMAGE)
model_validation_op = create_component_from_func(model_validation, base_image=BASE_IMAGE)

@dsl.pipeline(
    name='Term Deposit Prediction Pipeline',
    description='A pipeline to predict customer subscription with MLflow tracking.'
)
def term_deposit_pipeline(mlflow_tracking_uri: str = "http://atlantic-mlflow.default.svc.cluster.local:80"):
    """
    This is the main pipeline definition, orchestrating the steps.
    """
    # Orchestrate the data preparation step
    data_prep_task = data_prep_op()
    
    # Orchestrate the model training and MLflow tracking step
    model_train_task = model_train_op(
        data_path=data_prep_task.output,
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    
    # Orchestrate the model validation and reporting step
    model_validation_op(
        model_path=model_train_task.output,
        data_path=data_prep_task.output
    )

if __name__ == '__main__':
    # Compile the pipeline to a YAML file for Kubeflow.
    kfp.compiler.Compiler().compile(term_deposit_pipeline, 'term-deposit-pipeline.yaml')