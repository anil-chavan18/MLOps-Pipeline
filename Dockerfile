# Start with a clean Python image
FROM python:3.9-slim

# Set the working directory for the container
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install all the necessary Python dependencies for the pipeline components
# This ensures all components have the same consistent environment.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main pipeline script and data file into the image
COPY kubeflow_pipeline.py .
COPY large_bank_data.xlsm .

# This is the command that will be run when the container is executed.
# It compiles the pipeline into the final YAML file.
CMD ["python", "kubeflow_pipeline.py"]