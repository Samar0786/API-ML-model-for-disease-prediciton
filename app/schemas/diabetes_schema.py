from pydantic import BaseModel, ConfigDict, Field


class DiabetesInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    gender: str | int
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str | int
    bmi: float
    hba1c_level: float = Field(..., alias="HbA1c_level")
    blood_glucose_level: float