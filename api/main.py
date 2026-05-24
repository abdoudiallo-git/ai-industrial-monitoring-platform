from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
from src.inference.predictor import load_model, predict

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Industrial Monitoring API",
    description="API for predicting industrial machine failures using ML.",
    version="1.0.0"
)

# ── Load model once at startup ────────────────────────────────────────────────
model = load_model()

# ── Schemas ───────────────────────────────────────────────────────────────────
class SensorInput(BaseModel):
    type: str = Field(..., example="L", description="Machine type : L, M or H")
    air_temp: float = Field(..., example=300.0, description="Air temperature in Kelvin")
    process_temp: float = Field(..., example=310.0, description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., example=1500, description="Rotational speed in RPM")
    torque: float = Field(..., example=40.0, description="Torque in Nm")
    tool_wear: int = Field(..., example=100, description="Tool wear in minutes")

class PredictionOutput(BaseModel):
    prediction: int
    failure_probability: float
    status: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AI Industrial Monitoring API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict_failure(input_data: SensorInput):
    """
    Predict machine failure from sensor values.
    Returns prediction (0/1), failure probability and status.
    """
    try:
        data = input_data.model_dump()
        result = predict(model, data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))