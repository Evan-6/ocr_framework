import datetime
import json
import os
from contextlib import contextmanager

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    config_file = Column(String)
    config_overrides = Column(Text, default="{}")
    status = Column(String, default="pending")  # pending, running, done, failed, stopped
    pid = Column(Integer, nullable=True)
    metrics = Column(Text, default="{}")        # JSON
    out_dir = Column(String, nullable=True)     # where runs/<job_id> will be stored
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    @property
    def metrics_dict(self):
        try:
            return json.loads(self.metrics)
        except Exception:
            return {}

    @property
    def overrides_dict(self):
        try:
            return json.loads(self.config_overrides)
        except Exception:
            return {}

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    path = Column(String)
    num_samples = Column(Integer, default=0)
    chars = Column(String, default="")
    max_length = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

DB_PATH = os.environ.get("OCR_DB", "web_backend.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
