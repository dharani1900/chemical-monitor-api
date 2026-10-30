from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, EmailStr

# Auth Schemas
class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: Optional[str] = "user"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

# Detection Schemas
class DetectionOut(BaseModel):
    id: int
    user_id: int
    original_image_path: str
    visualization_image_path: Optional[str] = None
    prediction: str # 'leakage' or 'no_leakage'
    confidence: float
    mode: str # 'trained_model' or 'demo_mode'
    visualization_type: str
    severity: Optional[str] = None
    alarm_recommended: bool
    alarm_activated: bool
    model_version: str
    notes: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True

class DashboardStats(BaseModel):
    total_images: int
    leakage_predictions: int
    safe_predictions: int
    latest_detection: Optional[DetectionOut] = None
    alarm_status: str # 'active', 'silent', 'standby'
    model_status: str # 'trained', 'training', 'demo_mode', 'not_trained'
    detections_over_time: List[dict]
    prediction_breakdown: List[dict]

class ModelStatusOut(BaseModel):
    status: str # 'trained', 'training', 'demo_mode', 'not_trained'
    model_version: str
    is_custom_trained: bool
    last_trained_at: Optional[str] = None
    training_metrics: Optional[dict] = None

class DatasetValidationOut(BaseModel):
    is_valid: bool
    classes_found: List[str]
    train_counts: dict
    val_counts: dict
    test_counts: dict
    total_images: int
    message: str

class TrainingStartOut(BaseModel):
    job_id: str
    status: str
    message: str

class TrainingStatusOut(BaseModel):
    job_id: Optional[str] = None
    status: str
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Optional[Any] = None
    error_message: Optional[str] = None
