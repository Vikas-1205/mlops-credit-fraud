import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

# ---------- CONFIG ----------
BASELINE_F1 = 0.91
THRESHOLD = 0.91
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

# ---------- LOAD DATA ----------
print("📊 Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ---------- BALANCE DATA USING SMOTE ----------
print("⚖️ Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"✅ Before SMOTE: {sum(y==1)} fraud cases, {sum(y==0)} non-fraud")
print(f"✅ After SMOTE:  {sum(y_resampled==1)} fraud cases, {sum(y_resampled==0)} non-fraud")

# ---------- SPLIT DATA ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ---------- TRAIN MODEL ----------
print("🧠 Training model (Random Forest)...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
f1 = f1_score(y_test, preds)
acc = accuracy_score(y_test, preds)

print(f"✅ Model trained | F1-score: {f1:.4f} | Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, preds))

# ---------- MLFLOW LOGGING ----------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("CreditCardFraudDetection")

with mlflow.start_run():
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"📈 Baseline F1-score: {BASELINE_F1}")
    if f1 >= THRESHOLD:
        print("🚀 New model passed threshold! Registering for deployment...")
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "FraudDetectionModel"
        )
    else:
        print("❌ Model did not outperform baseline. Skipping deployment.")

print("🏁 Pipeline completed successfully.")
