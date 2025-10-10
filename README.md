# 💳 Credit Card Fraud Detection – End-to-End MLOps Pipeline

### 🚀 Overview
This project implements a **complete MLOps workflow** for detecting fraudulent credit card transactions using **MLflow**, **SMOTE**, and **FastAPI**.  

The goal is to build a **reproducible, automated machine learning pipeline** that:
- Handles class imbalance using **SMOTE**.
- Tracks all training runs and metrics using **MLflow**.
- Automatically registers the model only if performance exceeds a threshold (F1 ≥ 0.91).
- Deploys the best model via a **FastAPI REST endpoint**.

---

## ⚙️ Tech Stack
| Category | Tool |
|-----------|------|
| ML Framework | Scikit-learn |
| MLOps | MLflow |
| Deployment | FastAPI |
| Data Balancing | SMOTE (Imbalanced-Learn) |
| Language | Python 3.10+ |
| Dataset | Credit Card Fraud (Kaggle) |

---

## 🧠 Problem Statement
Credit card fraud detection is a highly **imbalanced classification problem**, where fraudulent transactions make up less than 0.2% of all transactions.  
The goal is to design a model that **detects fraudulent activity** while maintaining high precision and recall.

---

## 🔁 Workflow Architecture

        ┌────────────────────────┐
        │  Data Preprocessing    │
        │ (SMOTE Balancing Step) │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Model Training       │
        │ (Random Forest)        │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   MLflow Tracking      │
        │ (Metrics + Artifacts)  │
        └────────────┬───────────┘
                     │
       F1 < 0.91 ────┴──► Skip Deployment ❌
                     │
       F1 ≥ 0.91 ────┬──► Register Model 🚀
                     │
                     ▼
        ┌────────────────────────┐
        │ FastAPI Inference API  │
        │ (Real-Time Prediction) │
        └────────────────────────┘



---

## ⚡️ Improvements Implemented

🧩 1. **SMOTE Balancing**
Applied **Synthetic Minority Oversampling Technique (SMOTE)** to handle data imbalance.

| Metric | Before SMOTE | After SMOTE |
|---------|---------------|--------------|
| F1-score | 0.8571 | **0.9922** |
| Accuracy | 0.9996 | **0.9922** |
| Fraud Cases | 492 | **284,315 (balanced)** |

✅ After balancing, the model achieved much higher recall and F1-score.

---

🧩 2. **Conditional Deployment**
Only deploys (registers) the model if:
```python
if f1 >= 0.91:
    register_model_to_mlflow()

🧩 3. MLflow Integration

Tracks:
	•	Model parameters
	•	Accuracy & F1 metrics
	•	Model version
	•	Registry stage (Production)

✅ Each model version is stored for full reproducibility.

⸻

🧩 4. FastAPI Deployment

Provides a REST API endpoint /predict for real-time fraud prediction.

Example input:

{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536346,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}

Response:

{"fraudulent": false}

🧾 How to Run

1️⃣ Clone Repository

git clone https://github.com/<your-username>/mlops-credit-fraud.git
cd mlops-credit-fraud

2️⃣ Install Dependencies

pip3 install -r requirements.txt

3️⃣ Add Dataset

Download creditcard.csv from Kaggle and place it inside /data.

4️⃣ Run ML Pipeline

python3 model_pipeline.py

5️⃣ Run MLflow UI

mlflow ui

Visit http://127.0.0.1:5000

6️⃣ Run API

uvicorn app.main:app --reload

Open 👉 http://127.0.0.1:8000/docs

🧠 Key Learnings
	•	Automated ML pipeline creation with MLOps best practices.
	•	Hands-on experience in experiment tracking, model registry, and conditional deployment.
	•	Improved model fairness and recall using SMOTE.
	•	Deployment-ready API integration via FastAPI.

🏁 Final Outcome

F1-score: 0.9922 (Improved)
Automated Model Promotion to Production
Complete MLOps Lifecycle with API Integration
