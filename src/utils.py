import logging
import os

def setup_logging():
    """
    Set up logging configuration.
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("FoodGlucoseApp")
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler("logs/food_glucose_app.log")
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger