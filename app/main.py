from fastapi import FastAPI, HTTPException, Body
from datetime import datetime
from pydantic import BaseModel

# ==================== Diabetes Risk Prediction Service ====================

class NodeJsPredictionRequest(BaseModel):
    patient_id: str
    glucose_level: float

class NodeJsPredictionResponse(BaseModel):
    patient_id: str
    ml_predicted_risk_level: str
    risk_description: str

def predict_risk(glucose_level: float) -> str:
    if glucose_level < 70:
        return "low"
    elif 70 <= glucose_level < 140:
        return "normal"
    elif 140 <= glucose_level < 200:
        return "high"
    return "very high"

def get_risk_description(risk_level: str) -> str:
    descriptions = {
        "low": "Low risk of diabetes",
        "normal": "Normal glucose levels",
        "high": "Elevated risk of diabetes",
        "very high": "High risk of diabetes"
    }
    return descriptions.get(risk_level, "Unknown risk level")

# ==================== Food Glucose Prediction Service ====================

class FoodInput(BaseModel):
    food_name: str

def setup_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Load model and vectorizer (would normally be in separate files)
try:
    # These would be actual model files in a real implementation
    model = None
    vectorizer = None
    logger.info("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model or vectorizer file not found: {e}")
    raise FileNotFoundError("Ensure model files exist")

def predict_glucose(food_name: str, model, vectorizer) -> float:
    # Mock implementation - replace with actual model prediction
    return 42.0  # Example value

def get_diabetic_recommendation(glucose_content: float, food_name: str) -> dict:
    # Mock implementation
    return {
        "glycemic_load": "medium",
        "recommendation": "Moderate consumption",
        "details": f"{food_name} has moderate glucose content"
    }

# ==================== FastAPI App Setup ====================

app = FastAPI(
    title="Combined Diabetes Services API",
    description="API for diabetes risk prediction and food glucose content prediction",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== Diabetes Risk Endpoints ====================

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

# ==================== Food Prediction Endpoints ====================

@app.post("/api/v1/predict_food_glucose/")
async def predict_glucose_content(food_input: FoodInput):
    try:
        food_name = food_input.food_name.strip()
        if not food_name:
            raise ValueError("Food name cannot be empty.")
        
        glucose_content = predict_glucose(food_name, model, vectorizer)
        recommendation = get_diabetic_recommendation(glucose_content, food_name)
        glycemic_load = recommendation.get("glycemic_load")
        
        logger.info(f"Prediction for '{food_name}': {glucose_content:.2f} g/100g, GL: {glycemic_load}")
        return {
            "food_name": food_name,
            "glucose_content_g_per_100g": glucose_content,
            "glycemic_load": glycemic_load,
            "diabetic_recommendation": {
                "recommendation": recommendation["recommendation"],
                "details": recommendation["details"]
            }
        }
    except Exception as e:
        logger.error(f"Error processing request for '{food_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting glucose content: {e}")

# ==================== Common Endpoints ====================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": ["diabetes_risk", "food_glucose"]
    }

# ==================== Main Execution ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)