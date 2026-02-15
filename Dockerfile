FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY model_pipeline.py .

# Copy mlruns if available (for pre-trained models)
COPY mlruns/ ./mlruns/

# Expose port
EXPOSE 8000

# Set environment variables
ENV MLFLOW_TRACKING_URI=mlruns
ENV MODEL_NAME=FraudDetectionModel

# Run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
