from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.routes import heart, liver, diabetes
from app.core.model_loader import load_all_models

app = FastAPI(
    title="Smart Hospital AI Prediction API",
    version="1.0.0",
)

# CORS (IMPORTANT FOR FRONTEND)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")


@app.on_event("startup")
def startup_event():
    logger.info("🚀 Starting API...")
    load_all_models()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return JSONResponse({"status": "healthy"})


# ROUTES
app.include_router(heart.router, prefix="/predict")
app.include_router(liver.router, prefix="/predict")
app.include_router(diabetes.router, prefix="/predict")