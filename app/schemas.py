#app/schemas.py
from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field, validator  # Compatible with Pydantic v1 and v2

class RiskLevel(int, Enum):
    """
    Enum for diabetes risk levels.
    Matches the levels expected by the ML model and used in communication.
    0: No Diabetes (or very low risk based on glucose)
    1: Low Risk
    2: Medium Risk
    3: High Risk
    """
    no_diabetes = 0
    low_risk = 1
    medium_risk = 2
    high_risk = 3

class NodeJsPredictionRequest(BaseModel):
    """
    Request model for predictions initiated by the Node.js backend.
    'patient_id' here is the identifier used by Node.js (e.g., phone number or unique ID).
    """
    patient_id: str = Field(
        ...,
        description="Patient identifier from the calling system (e.g., phone number or unique ID from Node.js DB)",
        min_length=1
    )
    glucose_level: float = Field(
        ...,
        gt=0,
        description="Glucose level in mg/dL. Must be a positive number."
    )

    @validator('patient_id')
    def patient_id_must_not_be_empty(cls, value: str) -> str:
        if not value.strip(): # Ensure it's not just whitespace
            raise ValueError('patient_id must not be empty or whitespace')
        return value

class NodeJsPredictionResponse(BaseModel):
    """
    Response model for predictions sent back to the Node.js backend
    or for the direct prediction utility endpoint.
    """
    patient_id: str = Field(..., description="Patient identifier (echoed back from request or 'N/A' for direct prediction)")
    ml_predicted_risk_level: RiskLevel = Field(..., description="Diabetes risk level predicted by the ML model (0-3)")
    risk_description: Optional[str] = Field(None, description="Human-readable description of the risk level")

    @validator('ml_predicted_risk_level', pre=True)
    def validate_risk_level_int(cls, value: Union[int, RiskLevel]) -> RiskLevel:
        if isinstance(value, RiskLevel): # If it's already a RiskLevel enum, pass it through
            return value
        if isinstance(value, int):
            try:
                return RiskLevel(value) # Attempt to convert int to RiskLevel enum
            except ValueError:
                valid_values = [r.value for r in RiskLevel]
                raise ValueError(f"Invalid integer value '{value}' for RiskLevel. Must be one of {valid_values}.")
        raise TypeError(f"Invalid type for ml_predicted_risk_level: {type(value)}. Expected int or RiskLevel enum.")

    class Config:
        json_encoders = {
            RiskLevel: lambda v: v.value  # Ensure enums are serialized as their integer values
        }
        from_attributes = True  # For ORM compatibility