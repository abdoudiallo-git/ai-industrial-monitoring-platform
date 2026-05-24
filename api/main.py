from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
from src.inference.predictor import load_model, predict
from api.database import get_db, PredictionRecord

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
    id: int
    prediction: int
    failure_probability: float
    status: str
    created_at: str

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
def predict_failure(input_data: SensorInput, db: Session = Depends(get_db)):
    """
    Predict machine failure from sensor values.
    Stores the prediction in PostgreSQL and returns the result.
    """
    try:
        data = input_data.model_dump()
        result = predict(model, data)

        # Save to database
        record = PredictionRecord(
            machine_type=input_data.type,
            air_temp=input_data.air_temp,
            process_temp=input_data.process_temp,
            rotational_speed=input_data.rotational_speed,
            torque=input_data.torque,
            tool_wear=input_data.tool_wear,
            prediction=result["prediction"],
            failure_probability=result["failure_probability"],
            status=result["status"]
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "id": record.id,
            "prediction": result["prediction"],
            "failure_probability": result["failure_probability"],
            "status": result["status"],
            "created_at": str(record.created_at)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def get_predictions(limit: int = 10, db: Session = Depends(get_db)):
    """Get last predictions from database."""
    records = db.query(PredictionRecord).order_by(
        PredictionRecord.created_at.desc()
    ).limit(limit).all()

    return [
        {
            "id": r.id,
            "created_at": str(r.created_at),
            "machine_type": r.machine_type,
            "prediction": r.prediction,
            "failure_probability": r.failure_probability,
            "status": r.status
        }
        for r in records
    ]