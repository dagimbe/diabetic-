import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from src.model_training import predict_glucose, get_diabetic_recommendation
from src.utils import setup_logging

# Initialize FastAPI app
app = FastAPI(title="Food Glucose Predictor API", description="API for predicting glucose content and diabetic recommendations.")

# Set up logging
logger = setup_logging()

# Load model and vectorizer
try:
    with open("../food_glucose_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("../food_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model or vectorizer file not found: {e}")
    raise FileNotFoundError("Ensure food_glucose_model.pkl and food_vectorizer.pkl exist.")

# Define request body schema
class FoodInput(BaseModel):
    food_name: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
async def predict_glucose_content(food_input: FoodInput):
    """
    Predict glucose content and diabetic recommendation for a given food name.
    Args:
        food_input (FoodInput): JSON object with food_name field (e.g., {"food_name": "Injera"}).
    Returns:
        dict: Glucose content, glycemic load, and diabetic recommendation.
    """
    try:
        food_name = food_input.food_name.strip()
        if not food_name:
            raise ValueError("Food name cannot be empty.")
        
        # Predict glucose content
        glucose_content = predict_glucose(food_name, model, vectorizer)
        
        # Get diabetic recommendation
        recommendation = get_diabetic_recommendation(glucose_content, food_name)
        
        # Extract glycemic load
        glycemic_load = recommendation.get("glycemic_load")
        
        logger.info(f"Prediction for '{food_name}': {glucose_content:.2f} g/100g, GL: {glycemic_load}, {recommendation}")
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

if __name__ == "__main__":
    import uvicorn
    port = 8000
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as e:
        logger.error(f"Port {port} is in use, trying port 8001")
        uvicorn.run(app, host="0.0.0.0", port=8001)