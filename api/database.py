import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# ── SQLAlchemy setup ──────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── Model ─────────────────────────────────────────────────────────────────────
class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    machine_type = Column(String(1), nullable=False)
    air_temp = Column(Float, nullable=False)
    process_temp = Column(Float, nullable=False)
    rotational_speed = Column(Integer, nullable=False)
    torque = Column(Float, nullable=False)
    tool_wear = Column(Integer, nullable=False)
    prediction = Column(Integer, nullable=False)
    failure_probability = Column(Float, nullable=False)
    status = Column(String(10), nullable=False)

# ── Dependency ────────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()