import os
import json
import zipfile
import shutil
import uuid
import threading
import time
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODEL_DIR = os.path.join(PROJECT_DIR, "storage", "models")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

class TrainingManager:
    def __init__(self):
        self.active_job: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()

    def validate_dataset(self, target_dir: str = DATASET_DIR) -> Dict[str, Any]:
        """Validates the dataset folder structure and counts images per class."""
        splits = ["train", "val", "test"]
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        counts = {
            "train": {"leakage": 0, "no_leakage": 0},
            "val": {"leakage": 0, "no_leakage": 0},
            "test": {"leakage": 0, "no_leakage": 0},
        }

        total_images = 0
        classes_found = set()

        for split in splits:
            split_dir = os.path.join(target_dir, split)
            if not os.path.exists(split_dir):
                continue

            for cls_name in ["leakage", "no_leakage"]:
                cls_dir = os.path.join(split_dir, cls_name)
                if os.path.exists(cls_dir):
                    classes_found.add(cls_name)
                    for f in os.listdir(cls_dir):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in valid_extensions:
                            counts[split][cls_name] += 1
                            total_images += 1

        is_valid = (counts["train"]["leakage"] > 0) and (counts["train"]["no_leakage"] > 0)

        msg = "Dataset structure valid." if is_valid else "Dataset invalid: requires 'train/leakage' and 'train/no_leakage' with image files."

        return {
            "is_valid": is_valid,
            "classes_found": list(classes_found),
            "train_counts": counts["train"],
            "val_counts": counts["val"],
            "test_counts": counts["test"],
            "total_images": total_images,
            "message": msg
        }

    def process_dataset_zip(self, zip_filepath: str) -> Dict[str, Any]:
        """Safely extracts uploaded dataset ZIP file into dataset/ directory."""
        temp_extract = os.path.join(PROJECT_DIR, "storage", "temp_zip_extract")
        if os.path.exists(temp_extract):
            shutil.rmtree(temp_extract)
        os.makedirs(temp_extract, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                # Prevent path traversal vulnerabilities (Zip Slip)
                for member in zip_ref.namelist():
                    filename = os.path.basename(member)
                    if not filename:
                        continue
                    # Ensure no relative path navigation outside extract dir
                    target_path = os.path.abspath(os.path.join(temp_extract, member))
                    if not target_path.startswith(os.path.abspath(temp_extract)):
                        raise ValueError(f"Illegal path in zip file: {member}")

                zip_ref.extractall(temp_extract)

            # Look for dataset subfolders inside temp_extract
            extracted_items = os.listdir(temp_extract)
            source_dir = temp_extract
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract, extracted_items[0])):
                source_dir = os.path.join(temp_extract, extracted_items[0])

            validation = self.validate_dataset(source_dir)
            if not validation["is_valid"]:
                return validation

            # Clear current dataset and replace with extracted
            shutil.rmtree(DATASET_DIR, ignore_errors=True)
            shutil.copytree(source_dir, DATASET_DIR)
            shutil.rmtree(temp_extract, ignore_errors=True)

            return self.validate_dataset(DATASET_DIR)
        except Exception as e:
            if os.path.exists(temp_extract):
                shutil.rmtree(temp_extract, ignore_errors=True)
            return {
                "is_valid": False,
                "classes_found": [],
                "train_counts": {},
                "val_counts": {},
                "test_counts": {},
                "total_images": 0,
                "message": f"Error processing ZIP file: {str(e)}"
            }

    def start_training_job(self, epochs: int = 5, lr: float = 0.001) -> Dict[str, Any]:
        with self.lock:
            if self.active_job and self.active_job["status"] == "running":
                return {
                    "job_id": self.active_job["job_id"],
                    "status": "running",
                    "message": "A training job is already in progress."
                }

            job_id = str(uuid.uuid4())[:8]
            self.active_job = {
                "job_id": job_id,
                "status": "running",
                "current_epoch": 0,
                "total_epochs": epochs,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "confusion_matrix": None,
                "error_message": None,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Launch background thread for training
            thread = threading.Thread(target=self._run_training_thread, args=(epochs, lr))
            thread.daemon = True
            thread.start()

            return {
                "job_id": job_id,
                "status": "running",
                "message": f"Model training job {job_id} launched successfully!"
            }

    def _run_training_thread(self, epochs: int, lr: float):
        from train_model import train_and_evaluate

        def status_callback(epoch, total_epochs, train_loss, val_loss, val_acc):
            with self.lock:
                if self.active_job:
                    self.active_job["current_epoch"] = epoch
                    self.active_job["train_loss"] = round(train_loss, 4)
                    self.active_job["val_loss"] = round(val_loss, 4)
                    self.active_job["accuracy"] = round(val_acc, 4)

        try:
            metrics = train_and_evaluate(epochs=epochs, lr=lr, status_callback=status_callback)
            with self.lock:
                if self.active_job:
                    self.active_job["status"] = "completed"
                    self.active_job["current_epoch"] = epochs
                    self.active_job["accuracy"] = metrics.get("best_val_accuracy", 0.0)
                    self.active_job["precision"] = metrics.get("precision", 0.0)
                    self.active_job["recall"] = metrics.get("recall", 0.0)
                    self.active_job["f1_score"] = metrics.get("f1_score", 0.0)
                    self.active_job["confusion_matrix"] = metrics.get("confusion_matrix")

            # Reload trained model into ml_service
            from app.ml_service import ml_service
            ml_service.load_model()

        except Exception as e:
            print(f"Training thread exception: {e}")
            with self.lock:
                if self.active_job:
                    self.active_job["status"] = "failed"
                    self.active_job["error_message"] = str(e)

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            if self.active_job:
                return self.active_job.copy()

            # If no active job, check if model_info.json exists
            if os.path.exists(INFO_PATH):
                try:
                    with open(INFO_PATH, "r") as f:
                        info = json.load(f)
                    return {
                        "job_id": "latest_saved",
                        "status": "completed",
                        "current_epoch": len(info.get("history", [])),
                        "total_epochs": len(info.get("history", [])),
                        "train_loss": info.get("history", [{}])[-1].get("train_loss", 0.0) if info.get("history") else 0.0,
                        "val_loss": info.get("history", [{}])[-1].get("val_loss", 0.0) if info.get("history") else 0.0,
                        "accuracy": info.get("best_val_accuracy", 0.0),
                        "precision": info.get("precision", 0.0),
                        "recall": info.get("recall", 0.0),
                        "f1_score": info.get("f1_score", 0.0),
                        "confusion_matrix": info.get("confusion_matrix"),
                        "error_message": None
                    }
                except Exception:
                    pass

            return {
                "job_id": None,
                "status": "not_started",
                "current_epoch": 0,
                "total_epochs": 0,
                "train_loss": None,
                "val_loss": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "confusion_matrix": None,
                "error_message": None
            }

training_manager = TrainingManager()
