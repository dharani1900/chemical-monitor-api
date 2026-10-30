import os
import shutil
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from app.models import User
from app.schemas import DatasetValidationOut, TrainingStartOut, TrainingStatusOut
from app.auth import get_current_admin
from app.training_service import training_manager

router = APIRouter(prefix="/api/admin", tags=["Admin & Model Training"])

@router.post("/dataset/validate", response_model=DatasetValidationOut)
def validate_dataset_structure(current_admin: User = Depends(get_current_admin)):
    res = training_manager.validate_dataset()
    return res

@router.post("/dataset/upload-zip", response_model=DatasetValidationOut)
async def upload_dataset_zip(
    file: UploadFile = File(...),
    current_admin: User = Depends(get_current_admin)
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files containing labelled dataset folders are supported.")

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, file.filename)

    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        res = training_manager.process_dataset_zip(zip_path)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return res
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Dataset processing failed: {str(e)}")

@router.post("/training/start", response_model=TrainingStartOut)
def start_model_training(
    epochs: int = Form(5),
    lr: float = Form(0.001),
    current_admin: User = Depends(get_current_admin)
):
    val = training_manager.validate_dataset()
    if not val["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot start training: dataset is missing or invalid. Please upload a valid dataset first."
        )

    res = training_manager.start_training_job(epochs=epochs, lr=lr)
    return res

@router.get("/training/status", response_model=TrainingStatusOut)
def get_training_job_status(current_admin: User = Depends(get_current_admin)):
    return training_manager.get_status()

@router.get("/model/evaluation")
def get_model_evaluation_report(current_admin: User = Depends(get_current_admin)):
    status_data = training_manager.get_status()
    if status_data.get("status") != "completed":
        return {
            "message": "Model evaluation report available only after model training is completed.",
            "metrics": None
        }
    return {
        "message": "Model evaluation report generated successfully.",
        "metrics": status_data
    }
