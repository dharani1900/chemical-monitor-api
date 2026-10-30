from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from app.database import get_db
from app.models import User, DetectionRecord
from app.schemas import DashboardStats, DetectionOut
from app.auth import get_current_user
from app.ml_service import ml_service

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

LEAKAGE_DANGER_THRESHOLD = 0.52

@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Retrieve user-isolated records
    user_query = db.query(DetectionRecord).filter(DetectionRecord.user_id == current_user.id)

    total_images = user_query.count()
    danger_count = user_query.filter(
        DetectionRecord.prediction == "leakage",
        DetectionRecord.confidence >= LEAKAGE_DANGER_THRESHOLD
    ).count()
    safe_count = total_images - danger_count

    latest_record = user_query.order_by(DetectionRecord.timestamp.desc()).first()
    latest_out = DetectionOut.model_validate(latest_record) if latest_record else None

    # Alarm status determination: active only when danger threshold is met
    alarm_status = "silent"
    if latest_record and latest_record.prediction == "leakage" and latest_record.confidence >= LEAKAGE_DANGER_THRESHOLD:
        alarm_status = "active"

    # ML Model status
    model_status = ml_service.mode # 'trained_model', 'demo_mode'

    # Group detections over time (last 7 days)
    today = datetime.utcnow().date()
    timeline_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_str = day.strftime("%b %d")
        day_start = datetime.combine(day, datetime.min.time())
        day_end = datetime.combine(day, datetime.max.time())

        day_query = user_query.filter(DetectionRecord.timestamp >= day_start, DetectionRecord.timestamp <= day_end)
        day_total = day_query.count()
        day_danger = day_query.filter(
            DetectionRecord.prediction == "leakage",
            DetectionRecord.confidence >= LEAKAGE_DANGER_THRESHOLD
        ).count()
        day_safe = day_total - day_danger

        timeline_data.append({
            "date": day_str,
            "total": day_total,
            "leakage": day_danger,
            "safe": day_safe
        })

    prediction_breakdown = [
        {"name": "Safe Cabinets", "value": safe_count, "color": "#10b981"},
        {"name": "Danger Alerts", "value": danger_count, "color": "#ef4444"}
    ]

    return {
        "total_images": total_images,
        "leakage_predictions": danger_count,
        "safe_predictions": safe_count,
        "latest_detection": latest_out,
        "alarm_status": alarm_status,
        "model_status": model_status,
        "detections_over_time": timeline_data,
        "prediction_breakdown": prediction_breakdown
    }
