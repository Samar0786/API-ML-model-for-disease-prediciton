from fastapi import APIRouter, HTTPException
from app.schemas.heart_schema import HeartInput, HeartOutput
from app.services.heart_service import predict_heart

router = APIRouter(tags=["heart"])

@router.post("/heart", response_model=HeartOutput)
def heart_predict(input_data: HeartInput):
    try:
        # Mounted in app.main with prefix="/predict" so this becomes POST /predict/heart
        result = predict_heart(input_data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))