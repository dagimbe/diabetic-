# File: app/ml_model.py
import joblib
import numpy as np
from pathlib import Path
from app.schemas import RiskLevel # Assuming schemas.py is in the same 'app' directory

# --- Configuration for Model Path ---
# Determine the base directory of the project.
# This assumes 'app/ml_model.py' and the 'ml' folder are siblings under a project root.
# Example:
# project_root/
#   app/
#     ml_model.py
#     schemas.py
#     main.py
#   ml/
#     diabetes_risk_model.joblib
# If your structure is different, adjust BASE_DIR accordingly.
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ might not be defined in some contexts (e.g. some REPLs, frozen executables)
    # Fallback to current working directory, but this might be less reliable.
    BASE_DIR = Path.cwd()

MODEL_FILENAME = "diabetes_risk_model.joblib"
MODEL_PATH = BASE_DIR / "ml" / MODEL_FILENAME

# --- Load the trained model ---
model = None # Initialize model as None
try:
    if not MODEL_PATH.is_file(): # Use is_file() for clarity
        error_message = f"ML model file not found at expected path: {MODEL_PATH}. " \
                        f"Ensure the 'ml' directory exists at the project root ({BASE_DIR}) " \
                        f"and contains '{MODEL_FILENAME}'. " \
                        f"You might need to run the training script first."
        # This error is critical for the service's core functionality.
        # Raising an ImportError or RuntimeError at startup is appropriate.
        raise FileNotFoundError(error_message)
    model = joblib.load(MODEL_PATH)
    print(f"INFO: Successfully loaded ML model from {MODEL_PATH}")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load ML model from {MODEL_PATH}. Exception: {e}")
    # The application might not be able to function correctly without the model.
    # Depending on deployment, this might cause the app to fail to start, which is often desired.
    # For now, 'model' remains None, and predict_risk will raise an error.


def predict_risk(glucose_level: float) -> RiskLevel:
    """
    Predict diabetes risk from glucose level using the loaded model.
    Returns a RiskLevel enum member.
    """
    if model is None:
        # This indicates the model failed to load at application startup.
        raise RuntimeError("ML model is not available. Prediction cannot be performed. Check server logs for loading errors.")

    if not isinstance(glucose_level, (int, float)):
        raise ValueError("Glucose level must be a numeric value.")
    if glucose_level <= 0:
        # While the model might handle it, business logic usually expects positive glucose.
        raise ValueError("Glucose level must be a positive number.")

    try:
        # The model expects a 2D array-like structure as input, even for a single feature.
        input_data = np.array([[float(glucose_level)]]) # Ensure it's float for consistency
        # model.predict typically returns an array of predictions.
        # For a single input sample, it's an array with one element.
        prediction_int = model.predict(input_data)[0]

        # Convert the integer prediction to the RiskLevel enum.
        # This will raise a ValueError if prediction_int is not a valid enum value (0, 1, 2, 3).
        return RiskLevel(int(prediction_int))
    except ValueError as ve: # Catches issues from RiskLevel(int(prediction_int)) if value is out of enum range
        print(f"ERROR: Invalid risk level value '{prediction_int}' predicted by model for glucose {glucose_level}. Error: {ve}")
        # Decide on a fallback or re-raise. Re-raising makes the issue visible.
        raise RuntimeError(f"Prediction resulted in an invalid risk category: {prediction_int}. Check model output or enum definition.")
    except Exception as e:
        print(f"ERROR: Unexpected error during ML prediction for glucose level {glucose_level}: {e}")
        # General catch-all for other prediction issues.
        raise RuntimeError(f"An unexpected error occurred during the ML prediction process: {e}")


def get_risk_description(risk_level: RiskLevel) -> str:
    """
    Convert RiskLevel enum member to a human-readable description.
    """
    if not isinstance(risk_level, RiskLevel):
        # This shouldn't happen if type hints are respected, but good for robustness.
        return "Invalid risk level type provided."

    descriptions = {
        RiskLevel.no_diabetes: "No Diabetes Indicated by Glucose Level",
        RiskLevel.low_risk: "Low Diabetes Risk Indicated by Glucose Level",
        RiskLevel.medium_risk: "Medium Diabetes Risk Indicated by Glucose Level",
        RiskLevel.high_risk: "High Diabetes Risk Indicated by Glucose Level"
    }
    return descriptions.get(risk_level, f"Unknown Risk Level Value: {risk_level.value}")

# Example self-test (optional, for development)
if __name__ == "__main__":
    if model:
        print("\n--- Testing predict_risk function ---")
        test_levels = [50.0, 90.0, 130.0, 175.0, 230.0, -10.0, "abc"]
        for level in test_levels:
            print(f"\nTesting glucose level: {level}")
            try:
                risk = predict_risk(level)
                description = get_risk_description(risk)
                print(f"  Prediction -> RiskLevel Enum: {risk} (Value: {risk.value}), Name: {risk.name}")
                print(f"  Description: \"{description}\"")
            except (ValueError, RuntimeError) as e:
                print(f"  Error: {e}")
    else:
        print("ML Model was not loaded. Cannot run self-test for predict_risk.")