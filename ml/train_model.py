import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def load_data():
    """Load the synthetic dataset"""
    # Go up one level from ml/ to access data/
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes_dataset.csv')
    return pd.read_csv(data_path)

def train_and_save_model():
    """Train model and save it"""
    data = load_data()
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
    
    # Save model in the current directory (ml/)
    model_path = os.path.join(os.path.dirname(__file__), 'diabetes_risk_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model trained and saved successfully at: {model_path}")

if __name__ == "__main__":
    train_and_save_model()