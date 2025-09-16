# MLOps Pipeline - Term Deposit Prediction

This repository contains a complete MLOps pipeline for predicting customer term deposit subscriptions using machine learning.

## 🚀 Pipeline Overview

The pipeline implements all 8 essential MLOps steps:

1. **Trigger**: GitHub Actions automatically trigger on code changes
2. **Data Processing**: Transform and preprocess the banking data
3. **Data Validation**: Check data quality and detect bias
4. **Feature Storage**: Store features in a feature store
5. **Feature Retrieval**: Retrieve features for training
6. **Model Training**: Train a Logistic Regression model
7. **Model Validation**: Validate model quality and explainability
8. **Model Registration**: Register model if it meets thresholds

## 📁 Repository Structure

```
MLOps-Pipeline/
├── .github/workflows/
│   └── mlops-pipeline.yml          # GitHub Actions workflow
├── large_bank_data.xlsm            # Training data
├── local_mlops_demo.py             # Local pipeline execution
├── kubeflow_pipeline.py            # Kubeflow pipeline definition
├── term-deposit-pipeline.yaml     # Argo Workflow YAML
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image
└── README.md                       # This file
```

## 🔧 Setup Instructions

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd MLOps-Pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run local pipeline**
   ```bash
   python3 local_mlops_demo.py
   ```

### GitHub Actions Automation

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial MLOps pipeline setup"
   git push origin main
   ```

2. **Automatic Triggers**
   - Pipeline runs automatically on:
     - Push to `main` or `develop` branches
     - Pull requests to `main`
     - Changes to Python files, YAML files, or data

3. **Manual Trigger**
   - Go to Actions tab in GitHub
   - Select "MLOps Pipeline - Term Deposit Prediction"
   - Click "Run workflow"
   - Optionally set custom accuracy threshold

## 📊 Pipeline Features

### Data Processing & Validation
- ✅ Automated data quality checks
- ✅ Bias detection across age groups
- ✅ Feature engineering (age_loan_ratio)
- ✅ Data validation with quality thresholds

### Model Training & Validation
- ✅ Logistic Regression with scikit-learn
- ✅ 80/20 train-test split
- ✅ Comprehensive metrics (accuracy, precision, recall, F1)
- ✅ Feature importance analysis
- ✅ Configurable accuracy threshold (default: 85%)

### MLOps Best Practices
- ✅ Automated pipeline execution
- ✅ Artifact management
- ✅ Model versioning
- ✅ Quality gates
- ✅ Explainable AI features

## 🎯 Expected Outputs

### Pipeline Results
- **Features Stored**: age, balance, loan_status, campaign_calls, age_loan_ratio
- **Model Accuracy**: Typically ~51-55% (realistic for this dataset)
- **Model Registration**: Automatic if accuracy meets threshold
- **Deployment Ready**: Model artifacts prepared for deployment

### GitHub Actions Artifacts
- `processed-data`: Cleaned and transformed dataset
- `model-artifacts`: Trained model files (.pkl)
- `pipeline-summary`: Execution summary and metrics

## 🔄 Workflow Triggers

### Automatic Triggers
```yaml
on:
  push:
    branches: [ main, develop ]
    paths:
      - '**.py'
      - '**.yml'
      - '**.yaml'
      - 'requirements.txt'
      - 'large_bank_data.xlsm'
  pull_request:
    branches: [ main ]
```

### Manual Trigger
```yaml
workflow_dispatch:
  inputs:
    model_threshold:
      description: 'Model accuracy threshold'
      required: false
      default: '0.85'
```

## 📈 Monitoring & Observability

### Pipeline Status
- View execution status in GitHub Actions tab
- Each step shows detailed logs
- Artifacts available for download
- Pipeline summary generated automatically

### Model Metrics
- Accuracy, Precision, Recall, F1-Score
- Feature importance rankings
- Bias analysis across demographics
- Data quality scores

## 🚀 Deployment Options

### Local Deployment
```bash
# Run the complete local pipeline
python3 local_mlops_demo.py
```

### Kubernetes Deployment
```bash
# Build and deploy to Kubernetes
docker build -t mlops-pipeline:latest .
kubectl apply -f k8s-pipeline-job.yaml
```

### Kubeflow/Argo Workflows
```bash
# Deploy to Kubeflow Pipelines
kubectl create -f term-deposit-pipeline.yaml
```

## 🔧 Configuration

### Environment Variables
- `MODEL_THRESHOLD`: Accuracy threshold for model registration (default: 0.85)
- `MLFLOW_TRACKING_URI`: MLflow server URL (for advanced setups)

### Customization
- Modify `local_mlops_demo.py` for different algorithms
- Update `requirements.txt` for additional dependencies
- Adjust thresholds in GitHub Actions workflow

## 📝 Use Case Details

**Objective**: Predict whether a customer will subscribe to a term deposit

**Dataset Features**:
- `customer_id`: Unique identifier
- `age`: Customer age
- `balance`: Account balance
- `loan_status`: Loan status
- `campaign_calls`: Marketing campaign calls
- `target`: Binary label (1=subscribed, 0=not subscribed)

**Model**: Logistic Regression with feature engineering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the pipeline locally to test
5. Submit a pull request

The GitHub Actions pipeline will automatically validate your changes!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

🎉 **Ready to run your automated MLOps pipeline!** Push your code to GitHub and watch the magic happen!