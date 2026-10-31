import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Resolve candidate model paths
def resolve_file_path(filename: str) -> str:
    candidates = [
        os.path.join(PROJECT_DIR, "storage", "models", filename),
        os.path.join(BASE_DIR, "storage", "models", filename),
        os.path.join(os.getcwd(), "storage", "models", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

MODEL_PATH = os.getenv("MODEL_PATH") or resolve_file_path("best_model.pth")
INFO_PATH = os.getenv("MODEL_INFO_PATH") or resolve_file_path("model_info.json")

class MLService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ["leakage", "no_leakage"]
        self.mode = "demo_mode"
        self.model_version = "MobileNetV2-v1.0"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(INFO_PATH):
            try:
                with open(INFO_PATH, "r") as f:
                    info = json.load(f)
                    self.class_names = info.get("class_names", ["leakage", "no_leakage"])
                    self.model_version = info.get("model_version", "MobileNetV2-Custom-v1.0")

                model = models.mobilenet_v2(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features, len(self.class_names))
                )
                model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                model.to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.mode = "trained_model"
                print("Loaded trained PyTorch MobileNetV2 model successfully!")
            except Exception as e:
                print(f"Error loading trained model: {e}. Falling back to demo mode.")
                self.model = None
                self.mode = "demo_mode"
        else:
            print("No trained model found at storage/models/best_model.pth. App running in demo mode.")
            self.model = None
            self.mode = "demo_mode"

    def predict(self, image_path: str):
        """
        Runs model inference on uploaded image.
        Returns: dict with prediction ('leakage'|'no_leakage'), confidence, severity, mode, gradcam_heatmap.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # If trained model is available, perform real PyTorch inference & Grad-CAM
        if self.model is not None and self.mode == "trained_model":
            try:
                img_pil = Image.open(image_path).convert("RGB")
                input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

                features_blobs = []

                def forward_hook(module, inp, out):
                    features_blobs.append(out.detach().cpu())

                target_layer = self.model.features[-1]
                h = target_layer.register_forward_hook(forward_hook)

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])
                    pred_label = self.class_names[pred_idx]

                h.remove()

                # Compute Grad-CAM heatmap using forward activations and linear classifier weights
                cam_norm = None
                if len(features_blobs) > 0:
                    features = features_blobs[0][0].numpy()
                    try:
                        linear_layer = self.model.classifier[1]
                        if isinstance(linear_layer, nn.Sequential):
                            linear_layer = linear_layer[1]
                        weights = linear_layer.weight[pred_idx].detach().cpu().numpy()
                        cam = np.zeros(features.shape[1:], dtype=np.float32)
                        for i, w in enumerate(weights):
                            cam += w * features[i, :, :]
                        cam = np.maximum(cam, 0)
                        cam = cv2.resize(cam, (224, 224))
                        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    except Exception as cam_err:
                        print(f"Warning computing CAM: {cam_err}")
                        cam_norm = None

                severity = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
                if pred_label == "no_leakage":
                    severity = "None"

                return {
                    "prediction": pred_label,
                    "confidence": round(confidence, 4),
                    "severity": severity,
                    "mode": self.mode,
                    "model_version": self.model_version,
                    "gradcam_heatmap": cam_norm
                }
            except Exception as e:
                print(f"Prediction error in trained model: {e}. Falling back to demo inference.")

        # Demo fallback logic: inspect image or file traits to deliver consistent, deterministic demo predictions
        img_bgr = cv2.imread(image_path)
        if img_bgr is not None:
            # Color analysis: check average color or rust/yellow intensity in image to decide demo prediction
            avg_color = img_bgr.mean(axis=(0, 1))
            # If high red/yellow component, tag as leakage for demo testing
            is_demo_leakage = ("leak" in os.path.basename(image_path).lower()) or (avg_color[2] > 140 and avg_color[1] > 140)
        else:
            is_demo_leakage = True

        pred_label = "leakage" if is_demo_leakage else "no_leakage"
        confidence = 0.895 if is_demo_leakage else 0.942
        severity = "High" if is_demo_leakage else "None"

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "severity": severity,
            "mode": "demo_mode",
            "model_version": "MobileNetV2-Demo-v1.0",
            "gradcam_heatmap": None
        }

ml_service = MLService()
