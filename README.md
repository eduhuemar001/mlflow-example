# MLflow Example Project

This project demonstrates:
- MLflow Tracking (params/metrics/artifacts)
- Model logging and loading (runs:/ URIs)
- MLflow UI
- MLflow Projects (MLproject + conda env)
- Model serving

## Option A: Run directly (venv/pip)

```bash
python -m venv .venv
source .venv/bin/activate
pip install mlflow scikit-learn pandas numpy matplotlib

mlflow ui --port 5000
