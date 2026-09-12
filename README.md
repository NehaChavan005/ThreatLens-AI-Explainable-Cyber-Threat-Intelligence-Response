ThreatLens AI — Explainable Cyber Threat Detection & Response

An AI-powered intrusion detection system that detects malicious traffic, classifies the attack type, scores its risk, explains the decision, and recommends a defensive action — visualized on a live dashboard.

Architecture
Network Flow Features
        │
        ▼
Inference Service (XGBoost / FT-Transformer)
        │
        ▼
Attack Prediction + Confidence
        │
        ▼
Severity Scoring (Low / Medium / High / Critical)
        │
        ▼
SHAP Explainability (feature contribution)
        │
        ▼
Recommended Defensive Action (+ IBM Granite advisory)
        │
        ▼
Alert stored (SQLite) → Live Dashboard

Tech stack: FastAPI · XGBoost · PyTorch · SHAP · SQLAlchemy · SQLite · Chart.js · IBM Granite

Features
Detect — Classifies network traffic as normal or malicious
Classify — Identifies the specific attack type (DoS/DDoS, Infiltration, Botnet, Brute Force, SQL Injection, XSS, Port Scan, etc.)
Risk Score — Confidence score + severity level per alert
Explain — SHAP-based feature attribution showing why traffic was flagged
Recommend — Prioritized defensive actions per attack type
Visualize — Live dashboard with alerts, attack distribution, timeline, and top attackers
Working
Network flow data is sent to the prediction pipeline.
The model classifies it as benign or malicious, with a specific attack type.
A risk score and severity level are assigned.
SHAP identifies which features drove the decision.
A defensive action is recommended based on the attack type.
The alert appears on the dashboard with all of the above in one view.
Execution
Setup
bash
git clone https://github.com/NehaChavan005/ThreatLens-AI-Explainable-Cyber-Threat-Intelligence-Response.git
cd ThreatLens-AI-Explainable-Cyber-Threat-Intelligence-Response

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r app/requirements.txt
cp app/.env.example app/.env
Run
bash
uvicorn app.main:app --reload
Access
Endpoint	Purpose
http://127.0.0.1:8000/dashboard/	Analyst dashboard
http://127.0.0.1:8000/docs	API documentation
http://127.0.0.1:8000/health	System status
