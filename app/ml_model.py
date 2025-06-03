# app/ml_model.py
import joblib
import numpy as np
from app.schemas import RiskLevel

# Load the trained model
try:
    model = joblib.load("ml/diabetes_risk_model.joblib")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Run ml/train_model.py first")

def predict_risk(glucose_level: float) -> RiskLevel:
    """Predict diabetes risk from glucose level"""
    prediction = model.predict(np.array([[glucose_level]]))[0]
    return RiskLevel(int(prediction))

def get_risk_description(risk_level: RiskLevel) -> str:
    """Convert risk level to human-readable description"""
    return {
        RiskLevel.no_diabetes: "No Diabetes",
        RiskLevel.low_risk: "Diabetic, Low Risk",
        RiskLevel.medium_risk: "Diabetic, Medium Risk", 
        RiskLevel.high_risk: "Diabetic, High Risk"
    }.get(risk_level, "Unknown Risk Level")