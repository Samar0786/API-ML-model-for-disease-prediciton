from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging

from app.routes import heart, liver, diabetes

app = FastAPI(
    title="Smart Hospital AI Prediction API",
    description="APIs for disease risk prediction models",
    version="1.0.0",
)


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")


@app.on_event("startup")
def startup_event():
    logger.info("API Starting up...")


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return JSONResponse({"status": "healthy"})


# Register routes
app.include_router(heart.router)
app.include_router(liver.router, prefix="/predict")
app.include_router(diabetes.router, prefix="/predict")