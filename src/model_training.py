import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from src.utils import setup_logging

logger = setup_logging()

def train_model(df):
    """
    Train a Random Forest Regressor to predict glucose content.
    Args:
        df (pd.DataFrame): Dataset with food names and nutritional data.
    Returns:
        model: Trained model.
        vectorizer: Fitted TF-IDF vectorizer.
    """
    try:
        logger.info("Starting model training...")
        if df.empty or "Food_Name" not in df.columns or "Glucose_g_per_100g" not in df.columns:
            raise ValueError("Invalid dataset: missing required columns or empty.")

        # Feature extraction
        vectorizer = TfidfVectorizer(max_features=500, lowercase=True, stop_words="english")
        X = vectorizer.fit_transform(df["Food_Name"])
        y = df["Glucose_g_per_100g"]
        
        logger.info(f"Feature matrix shape: {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Feature matrix is empty after vectorization.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test, y_test)
        logger.info(f"Best model R^2 score: {score:.2f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return best_model, vectorizer
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def predict_glucose(food_name, model, vectorizer):
    """
    Predict glucose content for a given food name.
    Args:
        food_name (str): Name of the food.
        model: Trained model.
        vectorizer: Fitted TF-IDF vectorizer.
    Returns:
        float: Predicted glucose content (g/100g).
    """
    try:
        if not isinstance(food_name, str) or not food_name.strip():
            raise ValueError("Food name must be a non-empty string.")
        
        # Transform food name to vector
        food_vector = vectorizer.transform([food_name.lower()])
        logger.info(f"Food vector shape for '{food_name}': {food_vector.shape}")
        
        if food_vector.shape[1] == 0:
            raise ValueError(f"Vectorization produced an empty feature vector for '{food_name}'.")
        
        # Predict
        prediction = model.predict(food_vector)
        logger.info(f"Raw prediction for '{food_name}': {prediction}")
        
        if len(prediction) == 0:
            raise ValueError(f"Model prediction returned an empty array for '{food_name}'.")
        
        return round(float(prediction[0]), 2)
    
    except Exception as e:
        logger.error(f"Error predicting for '{food_name}': {e}")
        raise

def get_diabetic_recommendation(glucose_content, food_name):
    """
    Determine if a food is recommended for diabetic patients based on glucose content and glycemic load.
    Args:
        glucose_content (float): Predicted glucose content (g/100g).
        food_name (str): Name of the food.
    Returns:
        dict: Recommendation details with glycemic load.
    """
    try:
        # Estimate glycemic load based on typical values
        food_gi_ranges = {
            "Injera": (50, 57),  # Teff injera, low GI
            "Doro Wat": (40, 50),
            "Tibs": (0, 10),
            "Shiro": (45, 55),
            "Kitfo": (0, 10),
            "Misir Wat": (50, 60),
            "Gomen": (30, 40),
            "Ayib": (0, 10),
            "Teff Porridge": (50, 60),
            "Fitfit": (55, 65),
            "Pasta": (40, 50),
            "Croissant": (65, 75),
            "Baguette": (70, 80),
            "Pizza": (45, 55),
            "Roast Beef": (0, 10),
            "Mashed Potatoes": (80, 90),
            "Paella": (50, 60),
            "Tiramisu": (50, 60),
            "Schnitzel": (30, 40),
            "Risotto": (60, 70)
        }
        
        food_carb_ranges = {
            "Injera": (50, 60),
            "Doro Wat": (5, 15),
            "Tibs": (0, 5),
            "Shiro": (20, 30),
            "Kitfo": (0, 5),
            "Misir Wat": (25, 35),
            "Gomen": (5, 10),
            "Ayib": (0, 3),
            "Teff Porridge": (40, 50),
            "Fitfit": (45, 55),
            "Pasta": (65, 75),
            "Croissant": (40, 50),
            "Baguette": (50, 60),
            "Pizza": (30, 40),
            "Roast Beef": (0, 5),
            "Mashed Potatoes": (15, 25),
            "Paella": (20, 30),
            "Tiramisu": (30, 40),
            "Schnitzel": (10, 20),
            "Risotto": (25, 35)
        }
        
        # Use average GI and carbs for GL estimation
        gi = sum(food_gi_ranges.get(food_name, (50, 50))) / 2
        carbs = sum(food_carb_ranges.get(food_name, (30, 30))) / 2
        glycemic_load = round((carbs * gi) / 100, 2)
        
        # Recommendation logic
        if food_name.lower() == "injera":
            return {
                "recommendation": "Recommended",
                "details": "Teff injera has a low glycemic index (~50–57) and moderate glycemic load, suitable for diabetic patients in controlled portions.",
                "glycemic_load": glycemic_load
            }
        elif glycemic_load < 10 or glucose_content < 10:
            return {
                "recommendation": "Recommended",
                "details": f"Low glycemic load or glucose content ({glucose_content:.2f} g/100g), safe for diabetic patients.",
                "glycemic_load": glycemic_load
            }
        elif (10 <= glycemic_load <= 10) or (10 <= glucose_content <= 10):
            return {
                "recommendation": "Caution",
                "details": f"Moderate glycemic load or glucose content ({glucose_content:.2f} g/100g), consume in moderation for diabetic patients.",
                "glycemic_load": glycemic_load
            }
        else:
            return {
                "recommendation": "Not Recommended",
                "details": f"High glycemic load or glucose content ({glucose_content:.2f} g/100g), may cause blood sugar spikes.",
                "glycemic_load": glycemic_load
            }
    
    except Exception as e:
        logger.error(f"Error generating recommendation for '{food_name}': {e}")
        raise

if __name__ == "__main__":
    from src.data_generation import generate_food_dataset
    df = generate_food_dataset(1000)  # Smaller for testing
    model, vectorizer = train_model(df)
    with open("../food_glucose_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("../food_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Model and vectorizer saved.")