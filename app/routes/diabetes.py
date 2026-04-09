from fastapi import APIRouter, HTTPException
from app.schemas.diabetes_schema import DiabetesInput
from app.services.diabetes_service import predict_diabetes

router = APIRouter()

@router.post("/diabetes")
def predict(data: DiabetesInput):
    try:
        return predict_diabetes(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))