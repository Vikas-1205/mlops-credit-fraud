FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY model_pipeline.py .

# Copy exported model
COPY model/ ./model/

# Expose port (Render uses PORT env var)
EXPOSE 10000

# Set environment variables
ENV MODEL_PATH=model

# Run the API server — use PORT env var (Render sets this automatically)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
