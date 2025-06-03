# ml/data_generator.py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def generate_synthetic_data(num_samples=100000):
    """Generate synthetic diabetes dataset with glucose levels and risk labels"""
    np.random.seed(42)
    
    # Calculate sample size for each class (25% each)
    class_size = num_samples // 4
    
    # Class distributions
    # Class 0: No Diabetes (70-99 mg/dL fasting, <140 postprandial)
    glucose_class0 = np.random.normal(90, 10, class_size)
    glucose_class0 = np.clip(glucose_class0, 70, 140)
    
    # Class 1: Diabetic, Low Risk (100-150 mg/dL)
    glucose_class1 = np.random.normal(130, 15, class_size)
    glucose_class1 = np.clip(glucose_class1, 100, 180)
    
    # Class 2: Diabetic, Medium Risk (151-200 mg/dL)
    glucose_class2 = np.random.normal(175, 20, class_size)
    glucose_class2 = np.clip(glucose_class2, 150, 250)
    
    # Class 3: Diabetic, High Risk (>200 mg/dL)
    glucose_class3 = np.random.normal(230, 30, class_size)
    glucose_class3 = np.clip(glucose_class3, 200, 350)
    
    # Combine all classes
    glucose = np.concatenate([glucose_class0, glucose_class1, glucose_class2, glucose_class3])
    labels = np.concatenate([
        np.zeros(class_size),
        np.ones(class_size),
        np.ones(class_size) * 2,
        np.ones(class_size) * 3
    ])
    
    # Add some noise and shuffle
    glucose += np.random.normal(0, 5, num_samples)
    data = pd.DataFrame({'glucose_level': glucose, 'risk_level': labels})
    data = data.sample(frac=1).reset_index(drop=True)
    
    return data

def train_model(data):
    """Train and evaluate a RandomForest classifier"""
    X = data[['glucose_level']].values
    y = data['risk_level'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    # Get the project root directory (where diabetes_risk_system folder is)
    project_root = Path(__file__).parent.parent
    
    # Define paths
    data_path = project_root / 'data' / 'diabetes_dataset.csv'
    model_path = project_root / 'ml' / 'diabetes_risk_model.joblib'
    
    # Create data directory if it doesn't exist
    os.makedirs(project_root / 'data', exist_ok=True)
    
    # Generate and save synthetic data (100,000 samples)
    print("Generating 100,000 samples...")
    data = generate_synthetic_data(num_samples=100000)
    data.to_csv(data_path, index=False)
    print(f"Data saved to: {data_path}")
    
    # Verify data size
    print(f"\nData verification:")
    print(f"Total records: {len(data)}")
    print("Class distribution:")
    print(data['risk_level'].value_counts().sort_index())
    
    # Train and save model
    print("\nTraining model...")
    model = train_model(data)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")