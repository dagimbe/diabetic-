from app.ml_model import predict_risk, get_risk_description
import pytest

@pytest.mark.parametrize("glucose,expected", [
    (90, 0),    # No diabetes
    (130, 1),   # Low risk
    (175, 2),   # Medium risk
    (230, 3)    # High risk
])
def test_predict_risk(glucose, expected):
    """Test risk prediction"""
    assert predict_risk(glucose).value == expected

def test_risk_descriptions():
    """Test risk level descriptions"""
    assert get_risk_description(0) == "No Diabetes"
    assert get_risk_description(1) == "Diabetic, Low Risk"
    assert get_risk_description(2) == "Diabetic, Medium Risk"
    assert get_risk_description(3) == "Diabetic, High Risk"