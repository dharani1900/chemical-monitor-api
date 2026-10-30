import os
import cv2
import numpy as np
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "dataset")

def create_synthetic_cabinet_image(has_leakage: bool, index: int) -> np.ndarray:
    """
    Generates a realistic synthetic image of a closed chemical storage cabinet.
    Adds subtle visual anomalies (rust/corrosion around seals, venting marks) for leakage cases.
    """
    width, height = 400, 400
    # Background: Industrial lab wall / floor tile pattern
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (220, 225, 230) # Light grey lab wall

    # Wall tile grid lines
    for y in range(0, height, 50):
        cv2.line(img, (0, y), (width, y), (200, 205, 210), 1)
    for x in range(0, width, 50):
        cv2.line(img, (x, 0), (x, height), (200, 205, 210), 1)

    # Cabinet Body (Yellow/Grey Industrial Safety Cabinet)
    cab_color = (30, 180, 240) if index % 2 == 0 else (180, 180, 190) # Yellow or Metal grey BGR
    cv2.rectangle(img, (60, 40), (340, 360), cab_color, -1)
    cv2.rectangle(img, (60, 40), (340, 360), (40, 40, 40), 3) # Border

    # Cabinet Doors (Double door seam)
    cv2.line(img, (200, 40), (200, 360), (50, 50, 50), 3)

    # Door Handles & Locks
    cv2.rectangle(img, (185, 180), (195, 220), (30, 30, 30), -1)
    cv2.rectangle(img, (205, 180), (215, 220), (30, 30, 30), -1)

    # Hazard / Chemical Warning Label Sign
    cv2.rectangle(img, (90, 80), (170, 130), (255, 255, 255), -1)
    cv2.rectangle(img, (90, 80), (170, 130), (0, 0, 200), 2)
    cv2.putText(img, "DANGER", (98, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2)
    cv2.putText(img, "CHEMICALS", (95, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Ventilation Louvre Grills at bottom
    for y_vent in range(300, 340, 8):
        cv2.line(img, (80, y_vent), (160, y_vent), (40, 40, 40), 2)
        cv2.line(img, (240, y_vent), (320, y_vent), (40, 40, 40), 2)

    if has_leakage:
        # Add subtle visual anomaly around seam / louvre / hinges (discoloration, rust staining, vapor haze reflection)
        anomaly_x = random.choice([200, 160, 240])
        anomaly_y = random.choice([200, 310, 150])

        # Corrosion stain (brownish yellow)
        for r in range(25, 5, -5):
            cv2.circle(img, (anomaly_x + random.randint(-5, 5), anomaly_y + random.randint(-5, 5)), r, (20, 80, 160), -1)

        # Discoloration streak down the door seam
        cv2.line(img, (anomaly_x, anomaly_y), (anomaly_x, anomaly_y + 40), (10, 60, 140), 4)

    # Add light noise / texture
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img_noisy

def generate_demo_dataset():
    """Creates synthetic dataset structure for testing model training."""
    splits = ["train", "val", "test"]
    categories = ["leakage", "no_leakage"]

    counts = {
        "train": {"leakage": 25, "no_leakage": 25},
        "val": {"leakage": 8, "no_leakage": 8},
        "test": {"leakage": 8, "no_leakage": 8},
    }

    print(f"Generating synthetic dataset in {DATASET_DIR}...")
    for split in splits:
        for cat in categories:
            cat_dir = os.path.join(DATASET_DIR, split, cat)
            os.makedirs(cat_dir, exist_ok=True)

            num_imgs = counts[split][cat]
            has_leakage = (cat == "leakage")

            for i in range(num_imgs):
                img = create_synthetic_cabinet_image(has_leakage, i)
                filename = f"cabinet_{split}_{cat}_{i+1:03d}.jpg"
                filepath = os.path.join(cat_dir, filename)
                cv2.imwrite(filepath, img)

    print("Synthetic dataset successfully generated!")

if __name__ == "__main__":
    generate_demo_dataset()
