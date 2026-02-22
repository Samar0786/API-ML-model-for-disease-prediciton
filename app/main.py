from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging

from app.routes import heart

app = FastAPI(
    title="Smart Hospital AI Prediction API",
    description="APIs for disease risk prediction models",
    version="1.0.0"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

@app.on_event("startup")
def startup_event():
    logger.info("API Starting up...")

@app.get("/health")
def health_check():
    return JSONResponse({"status": "healthy"})

app.include_router(heart.router)