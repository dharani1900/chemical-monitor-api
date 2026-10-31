import os
import uuid
import shutil
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from PIL import Image

from app.database import get_db
from app.models import User, DetectionRecord
from app.schemas import DetectionOut
from app.auth import get_current_user
from app.ml_service import ml_service
from app.visualization import generate_gas_cloud_visualization

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
VISUALIZATIONS_DIR = os.path.join(STORAGE_DIR, "visualizations")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

router = APIRouter(prefix="/api/detections", tags=["Detections"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LEAKAGE_DANGER_THRESHOLD = 0.525

@router.post("/analyze", response_model=DetectionOut)
async def analyze_image(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file selected or invalid filename."
        )

    # Validate extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file format '{file_ext}'. Only JPG, JPEG, and PNG images are supported."
        )

    # Save uploaded image to storage/uploads
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOADS_DIR, filename)

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Check file size (0-byte and Max 10MB)
    file_size = os.path.getsize(upload_path)
    if file_size == 0:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty (0 bytes)."
        )

    if file_size > 10 * 1024 * 1024:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds maximum limit of 10MB."
        )

    # Verify image integrity and decodability
    try:
        with Image.open(upload_path) as img:
            img.verify()
    except Exception:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or unreadable image file. Please provide a valid JPG or PNG image."
        )

    # Perform PyTorch ML model inference
    try:
        res = ml_service.predict(upload_path)
    except Exception as e:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    prediction = res["prediction"] # 'leakage' or 'no_leakage'
    confidence = float(res["confidence"])
    mode = res["mode"]
    model_version = res["model_version"]
    gradcam_heatmap = res.get("gradcam_heatmap")

    # 52.5% Confidence Threshold strictly evaluated on the leakage class
    is_danger = (prediction == "leakage" and confidence >= LEAKAGE_DANGER_THRESHOLD)
    alarm_recommended = is_danger

    # Operational severity: only elevated if danger is detected
    if is_danger:
        severity = "High" if confidence > 0.85 else "Medium"
    else:
        severity = "None"

    # Generate Visualization Overlay
    vis_filename = f"vis_{filename}"
    vis_path = os.path.join(VISUALIZATIONS_DIR, vis_filename)

    try:
        generate_gas_cloud_visualization(
            image_path=upload_path,
            output_path=vis_path,
            is_leakage=is_danger,
            confidence=confidence,
            gradcam_heatmap=gradcam_heatmap if is_danger else None
        )
        rel_vis_path = f"/storage/visualizations/{vis_filename}"
    except Exception as e:
        print(f"[-] Visualization generation warning: {e}")
        rel_vis_path = f"/storage/uploads/{filename}"

    # Relative paths for frontend serving
    rel_upload_path = f"/storage/uploads/{filename}"

    # Create DB Record
    record = DetectionRecord(
        user_id=current_user.id,
        original_image_path=rel_upload_path,
        visualization_image_path=rel_vis_path,
        prediction=prediction,
        confidence=confidence,
        mode=mode,
        visualization_type="gradcam_or_simulated" if is_danger else "none",
        severity=severity,
        alarm_recommended=alarm_recommended,
        alarm_activated=alarm_recommended, # Trigger browser alarm on client side ONLY if danger threshold is met
        model_version=model_version,
        notes=notes,
        timestamp=datetime.utcnow()
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record

@router.get("", response_model=List[DetectionOut])
def list_detections(
    search: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None), # 'leakage' or 'no_leakage'
    sort: Optional[str] = Query("newest"), # 'newest' or 'oldest'
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(DetectionRecord).filter(DetectionRecord.user_id == current_user.id)

    if status_filter == "leakage":
        # Danger alerts: leakage prediction with confidence >= 80%
        query = query.filter(
            DetectionRecord.prediction == "leakage",
            DetectionRecord.confidence >= LEAKAGE_DANGER_THRESHOLD
        )
    elif status_filter == "no_leakage":
        # Safe cabinets: normal prediction OR leakage below 80% confidence
        query = query.filter(
            or_(
                DetectionRecord.prediction == "no_leakage",
                (DetectionRecord.prediction == "leakage") & (DetectionRecord.confidence < LEAKAGE_DANGER_THRESHOLD)
            )
        )

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (DetectionRecord.notes.like(search_pattern)) |
            (DetectionRecord.prediction.like(search_pattern)) |
            (DetectionRecord.model_version.like(search_pattern))
        )

    if sort == "oldest":
        query = query.order_by(DetectionRecord.timestamp.asc())
    else:
        query = query.order_by(DetectionRecord.timestamp.desc())

    return query.all()

@router.get("/{detection_id}", response_model=DetectionOut)
def get_detection(
    detection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    record = db.query(DetectionRecord).filter(
        DetectionRecord.id == detection_id,
        DetectionRecord.user_id == current_user.id
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail="Detection record not found.")
    return record

@router.delete("/{detection_id}", status_code=204)
def delete_detection(
    detection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    record = db.query(DetectionRecord).filter(
        DetectionRecord.id == detection_id,
        DetectionRecord.user_id == current_user.id
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail="Detection record not found.")

    # Remove stored image files if they exist locally
    orig_abs = os.path.join(BASE_DIR, record.original_image_path.lstrip("/"))
    vis_abs = os.path.join(BASE_DIR, record.visualization_image_path.lstrip("/")) if record.visualization_image_path else None

    if os.path.exists(orig_abs):
        try: os.remove(orig_abs)
        except Exception: pass

    if vis_abs and os.path.exists(vis_abs):
        try: os.remove(vis_abs)
        except Exception: pass

    db.delete(record)
    db.commit()
    return None
