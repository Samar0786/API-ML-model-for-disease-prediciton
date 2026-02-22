from fastapi import APIRouter, HTTPException
from app.schemas.heart_schema import HeartInput, HeartOutput
from app.services.heart_service import predict_heart

router = APIRouter(prefix="/predict/heart", tags=["heart"])

@router.post("", response_model=HeartOutput)
def heart_predict(input_data: HeartInput):
    try:
        result = predict_heart(input_data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))