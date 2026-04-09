from fastapi import APIRouter, HTTPException
from app.schemas.liver_schema import LiverInput
from app.services.liver_service import predict_liver

router = APIRouter()

@router.post("/liver")
def predict(data: LiverInput):
    try:
        return predict_liver(data.dict(by_alias=True))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))