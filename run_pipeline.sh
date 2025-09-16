#!/bin/bash

# --- User-configurable variables ---
PIPELINE_NAME="term-deposit-pipeline"
EXPERIMENT_NAME="Term Deposit Prediction"
DOCKER_IMAGE_NAME="mlops-pipeline:latest"
MLFLOW_TRACKING_URI="http://atlantic-mlflow.default.svc.cluster.local:80"

# ==============================================================================
# Step 7 & 8: Continuous Integration and Continuous Delivery
# This script automates the entire MLOps workflow.
# ==============================================================================

echo "🚀 Step 7 (CI): Building Docker image..."
# This command ensures all pipeline dependencies are packaged into a reproducible container image.
docker build -t ${DOCKER_IMAGE_NAME} .
if [ $? -ne 0 ]; then
    echo "❌ Docker image build failed."
    exit 1
fi
echo "✅ Docker image built successfully."

echo "📄 Compiling Kubeflow pipeline..."
# This command runs the Python script inside the Docker container to generate the final YAML manifest.
docker run --rm -v $(pwd):/app -w /app ${DOCKER_IMAGE_NAME} python kubeflow_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Pipeline compilation failed."
    exit 1
fi
echo "✅ Pipeline compiled to term-deposit-pipeline.yaml."

echo "📤 Step 8 (CD): Submitting pipeline to Kubeflow..."
# This command deploys the pipeline to your Kubeflow cluster on Rancher Desktop.
kfp run create \
--experiment-name ${EXPERIMENT_NAME} \
--pipeline-package ${PIPELINE_NAME}.yaml \
--arguments mlflow_tracking_uri=${MLFLOW_TRACKING_URI}

echo "✅ Pipeline submitted. Check the Kubeflow and MLflow UIs for the results."
echo "🎉 All steps completed successfully!"