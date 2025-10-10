from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load latest model version
model = mlflow.pyfunc.load_model("models:/FraudDetectionModel/1")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Get model’s expected feature names
    model_input_example = model.metadata.get_input_schema().input_names() if hasattr(model.metadata, "get_input_schema") else None

    # If model metadata unavailable, use dataset columns
    expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    # Fill missing columns with zeros
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[expected_columns]

    prediction = model.predict(df)[0]
    return {"fraudulent": bool(prediction)}
