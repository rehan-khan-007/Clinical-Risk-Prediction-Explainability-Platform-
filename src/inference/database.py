"""
Database layer. Defaults to a local SQLite file for development; set
DATABASE_URL to a postgres:// connection string for production. Schema is
identical either way since both are driven through SQLAlchemy.
"""
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """
    One row per prediction request. `consent_given` must be true for a row
    to be written at all — enforced in the API layer, not just recorded
    here — so every stored row reflects a real, explicit user choice to
    have their (anonymous) inputs kept.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    consent_given = Column(Boolean, nullable=False)
    nickname = Column(String, nullable=True)  # optional, never required

    # Input features (raw units — mmHg, kg, years — as submitted)
    inputs = Column(JSON, nullable=False)

    # Model output
    model_version = Column(String, nullable=False)
    risk_probability = Column(Float, nullable=False)
    risk_class = Column(String, nullable=False)  # "high" | "low"
    threshold_used = Column(Float, nullable=False)

    # Top SHAP contributions for this prediction, for the explanation UI
    top_contributions = Column(JSON, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
