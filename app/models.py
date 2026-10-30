import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user") # 'admin' or 'user'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    detections = relationship("DetectionRecord", back_populates="user", cascade="all, delete-orphan")

class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_image_path = Column(String, nullable=False)
    visualization_image_path = Column(String, nullable=True)
    prediction = Column(String, nullable=False) # 'leakage' or 'no_leakage'
    confidence = Column(Float, nullable=False)
    mode = Column(String, default="demo_mode") # 'trained_model' or 'demo_mode'
    visualization_type = Column(String, default="gradcam_or_simulated") # 'gradcam_or_simulated' or 'none'
    severity = Column(String, nullable=True) # 'High', 'Medium', 'Low', or 'None'
    alarm_recommended = Column(Boolean, default=False)
    alarm_activated = Column(Boolean, default=False)
    model_version = Column(String, default="MobileNetV2-v1.0")
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="detections")

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, default="pending") # 'pending', 'running', 'completed', 'failed'
    epochs = Column(Integer, default=5)
    current_epoch = Column(Integer, default=0)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    confusion_matrix = Column(Text, nullable=True) # JSON string representation
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
