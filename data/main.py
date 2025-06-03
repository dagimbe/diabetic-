# app/main.py
from fastapi import FastAPI, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from datetime import datetime
from app.schemas import NodeJsPredictionRequest, NodeJsPredictionResponse
from app.ml_model import predict_risk, get_risk_description

app = FastAPI(
    title="Diabetes Risk Prediction ML Service",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/api/v1/predict_risk_for_nodejs/", response_model=NodeJsPredictionResponse)
async def predict_for_nodejs(payload: NodeJsPredictionRequest = Body(...)):
    try:
        risk_level = predict_risk(payload.glucose_level)
        description = get_risk_description(risk_level)
        return NodeJsPredictionResponse(
            patient_id=payload.patient_id,
            ml_predicted_risk_level=risk_level,
            risk_description=description
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }