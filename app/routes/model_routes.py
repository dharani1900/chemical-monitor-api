from fastapi import APIRouter
from app.ml_service import ml_service
from app.schemas import ModelStatusOut

router = APIRouter(prefix="/api/model", tags=["Model Info"])

@router.get("/status", response_model=ModelStatusOut)
def get_model_info():
    is_custom = (ml_service.mode == "trained_model")
    return {
        "status": ml_service.mode,
        "model_version": ml_service.model_version,
        "is_custom_trained": is_custom,
        "last_trained_at": None,
        "training_metrics": None
    }

@router.post("/reload")
def reload_model_weights():
    ml_service.load_model()
    return {
        "status": ml_service.mode,
        "model_version": ml_service.model_version,
        "message": f"Model weights reloaded. Mode: {ml_service.mode}"
    }
