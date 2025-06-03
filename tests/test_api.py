from fastapi.testclient import TestClient
from data.main import app
import pytest

# Initialize TestClient correctly
client = TestClient(app)

def test_read_main():
    """Test root endpoint returns 404"""
    response = client.get("/")
    assert response.status_code == 404

def test_predict_endpoint():
    """Test prediction endpoint"""
    response = client.get("/predict/?glucose_level=120")
    assert response.status_code == 200
    assert "risk_level" in response.json()
    assert "description" in response.json()

def test_create_patient():
    """Test patient creation"""
    patient_data = {
        "name": "Test Patient",
        "age": 30,
        "gender": "male",
        "email": "test@example.com",
        "phone": "1234567890",
        "emergency_contact": {
            "name": "Emergency Contact",
            "phone": "0987654321"
        }
    }
    response = client.post("/patients/", json=patient_data)
    assert response.status_code == 201
    assert "id" in response.json()
    