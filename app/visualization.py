import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_gas_cloud_visualization(
    image_path: str,
    output_path: str,
    is_leakage: bool = True,
    confidence: float = 0.85,
    gradcam_heatmap: np.ndarray = None
) -> str:
    """
    Generates an explainable AI Grad-CAM / simulated gas cloud visualization overlay.
    Preserves original image, adds semi-transparent multi-colored gas plume (blue->cyan->yellow->orange->red),
    draws an intensity legend, and burns in an explicit scientific disclaimer banner.
    """
    # Read original image using OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image from {image_path}")

    height, width = img_bgr.shape[:2]

    if not is_leakage:
        # For safe (no leakage), output original image with green "SAFE CABINET" banner
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, height - 40), (width, height), (34, 197, 94), -1) # Green bar
        combined = cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0)
        cv2.putText(
            combined,
            "SAFE STATUS - NO LEAKAGE INDICATORS DETECTED",
            (15, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imwrite(output_path, combined)
        return output_path

    # If leakage predicted: generate simulated gas cloud plume
    if gradcam_heatmap is not None:
        # Resize Grad-CAM heatmap to original image size
        heatmap_resized = cv2.resize(gradcam_heatmap, (width, height))
        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    else:
        # Create a synthetic Gaussian plume centered on cabinet seam/door area
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        xx, yy = np.meshgrid(x, y)
        # Center plume slightly off-center (typical cabinet hinge/latch area)
        heatmap_norm = np.exp(-((xx - 0.2)**2 / 0.8 + (yy + 0.1)**2 / 0.5))
        heatmap_norm = (heatmap_norm - heatmap_norm.min()) / (heatmap_norm.max() - heatmap_norm.min())

    # Apply smooth thresholding to create plume shape
    plume_mask = np.clip((heatmap_norm - 0.2) / 0.8, 0, 1)

    # Map plume intensity to multi-color gradient (JET / TURBO colormap: Blue -> Cyan -> Yellow -> Orange -> Red)
    heatmap_uint8 = np.uint8(255 * plume_mask)
    color_plume = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blending with original image using opacity mask
    alpha_channel = (plume_mask * 0.65).astype(np.float32) # max 65% opacity
    alpha_3d = np.dstack([alpha_channel, alpha_channel, alpha_channel])

    blended = (color_plume.astype(np.float32) * alpha_3d + img_bgr.astype(np.float32) * (1.0 - alpha_3d)).astype(np.uint8)

    # Add plume contour outline for visual clarity
    contours, _ = cv2.findContours(heatmap_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)

    # Add Intensity Legend Bar (Bottom Right)
    legend_w, legend_h = 160, 20
    legend_x, legend_y = width - legend_w - 20, height - 70
    if legend_y > 40 and legend_x > 20:
        cv2.rectangle(blended, (legend_x - 10, legend_y - 25), (legend_x + legend_w + 10, legend_y + legend_h + 20), (15, 23, 42), -1)
        cv2.putText(blended, "Simulated Risk Density", (legend_x, legend_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (226, 232, 240), 1, cv2.LINE_AA)

        # Color bar
        for i in range(legend_w):
            val = int(255 * (i / legend_w))
            color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            cv2.line(blended, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_h), color.tolist(), 1)

        cv2.putText(blended, "Low", (legend_x, legend_y + legend_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(blended, "High", (legend_x + legend_w - 25, legend_y + legend_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

    # Add Disclaimer Banner at bottom (Red warning banner)
    banner_height = 40
    banner = np.zeros((banner_height, width, 3), dtype=np.uint8)
    banner[:] = (229, 57, 53) # Danger Red (BGR)

    cv2.putText(
        banner,
        "AI-SIMULATED LEAKAGE VISUALIZATION -- NOT A REAL GAS MEASUREMENT",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    final_img = np.vstack([blended, banner])
    cv2.imwrite(output_path, final_img)
    return output_path
