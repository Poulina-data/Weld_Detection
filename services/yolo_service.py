import os
import time
import logging
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger("yolo_vision_studio.service")

# High-contrast color palette for weld defects and generic classes (BGR format for OpenCV)
CLASS_PALETTE_BGR = {
    "good weld": (80, 200, 40),    # Vibrant Emerald Green
    "bad weld": (40, 40, 235),      # Bold Crimson Red
    "defect": (20, 165, 255),       # Bright Amber / Orange
}

# Fallback dynamic high-contrast colors (BGR) for arbitrary classes
DYNAMIC_PALETTE = [
    (235, 120, 40),   # Cyan/Azure
    (40, 220, 235),   # Yellow
    (220, 40, 200),   # Magenta/Violet
    (40, 180, 120),   # Mint
    (250, 180, 40),   # Sky Blue
    (140, 60, 240),   # Purple
    (40, 140, 255),   # Coral
    (180, 220, 50),   # Lime
]


class YOLOService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.class_names = {}
        self.is_loaded = False
        self.error_message = None
        self.model_name = "Unknown"
        self.is_custom_weld_model = False

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load YOLO weights from local path or Ultralytics model repo.
        """
        self.model_path = model_path
        self.model_name = Path(model_path).name if model_path else "yolo11n.pt"

        logger.info(f"Initializing YOLO Model from: {model_path} on device: {self.device}")

        # Check if local file was requested
        p = Path(model_path) if model_path else None
        if p and not p.exists() and not model_path.endswith((".pt", ".onnx", ".engine")):
            self.is_loaded = False
            self.error_message = f"Configured model file '{model_path}' does not exist on disk."
            logger.error(self.error_message)
            return False

        try:
            self.model = YOLO(model_path)
            # Warm up class names
            self.class_names = getattr(self.model, "names", {})
            self.is_loaded = True
            self.error_message = None

            # Detect if this is the weld defect model
            classes_lower = [str(name).lower() for name in self.class_names.values()]
            if any("weld" in name or "defect" in name for name in classes_lower):
                self.is_custom_weld_model = True

            logger.info(f"Model '{self.model_name}' loaded successfully with {len(self.class_names)} classes.")
            return True
        except Exception as e:
            self.is_loaded = False
            self.error_message = f"Failed to load YOLO model: {str(e)}"
            logger.exception(self.error_message)
            return False

    def get_class_color(self, class_name: str, class_id: int) -> tuple[int, int, int]:
        """
        Get BGR color for class name or fallback to index palette.
        """
        clean_name = str(class_name).strip().lower()
        if clean_name in CLASS_PALETTE_BGR:
            return CLASS_PALETTE_BGR[clean_name]
        return DYNAMIC_PALETTE[class_id % len(DYNAMIC_PALETTE)]

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640
    ) -> dict:
        """
        Run inference and generate annotated image + structured detection metadata.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.error_message or "YOLO model is not loaded.")

        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise ValueError("Input image is invalid or empty.")

        h_orig, w_orig = image_bgr.shape[:2]

        # Time the inference precisely
        t_start = time.perf_counter()
        results = self.model.predict(
            source=image_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )
        inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        detections = []
        class_counts = {}

        # Work on a copy of the image for annotation
        annotated = image_bgr.copy()

        # Scale line thickness and font according to image resolution
        base_scale = max(w_orig, h_orig) / 1000.0
        line_thickness = max(2, int(2.5 * base_scale))
        font_scale = max(0.5, 0.65 * base_scale)
        font_thickness = max(1, int(1.8 * base_scale))

        if results and len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy]

                # Constrain coordinates to image boundaries
                x1 = max(0, min(x1, w_orig - 1))
                y1 = max(0, min(y1, h_orig - 1))
                x2 = max(0, min(x2, w_orig - 1))
                y2 = max(0, min(y2, h_orig - 1))

                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = self.class_names.get(cls_id, f"Class {cls_id}")

                # Update per-class counter
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                # Colors
                color_bgr = self.get_class_color(cls_name, cls_id)
                color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"

                detections.append({
                    "id": idx + 1,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "confidence_pct": f"{conf * 100:.2f}%",
                    "bbox": [x1, y1, x2, y2],
                    "color": color_hex
                })

                # --- High-Contrast Professional Box Drawing ---
                # 1. Main bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, line_thickness)

                # 2. Sleek corner accent ticks
                corner_len = min(int((x2 - x1) * 0.2), int((y2 - y1) * 0.2), 25)
                if corner_len > 6:
                    accent_thick = line_thickness + 2
                    # Top-left
                    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), accent_thick)
                    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), accent_thick)
                    # Bottom-right
                    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), accent_thick)
                    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), accent_thick)

                # 3. Label Badge Header with contrasting solid background
                label_text = f"{cls_name} {conf * 100:.1f}%"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )

                # Position badge slightly above or inside box if at top border
                badge_y1 = max(0, y1 - label_h - baseline - 8)
                badge_y2 = badge_y1 + label_h + baseline + 8
                badge_x2 = min(w_orig - 1, x1 + label_w + 14)

                # Filled background rectangle for text
                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color_bgr, -1)
                # Subtle border around badge
                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)

                # Crisp white text with dark outline
                text_org = (x1 + 7, badge_y2 - baseline - 4)
                cv2.putText(
                    annotated,
                    label_text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

        # ---- Determine Overall Inspection Verdict ----
        verdict = self._compute_verdict(detections)

        return {
            "annotated_image": annotated,
            "detections": detections,
            "detection_count": len(detections),
            "class_counts": class_counts,
            "unique_classes_count": len(class_counts),
            "inference_time_ms": inference_time_ms,
            "verdict": verdict,
            "original_dims": {"width": w_orig, "height": h_orig}
        }

    def _compute_verdict(self, detections: list) -> dict:
        """
        Compute high-level inspection verdict based on detected objects.
        """
        if not detections:
            return {
                "status": "clean",
                "label": "No Objects Detected",
                "color": "#94a3b8",
                "badge": "neutral",
                "summary": "No target features detected at current confidence threshold."
            }

        # If custom weld model is in use:
        # Class 0: Bad Weld, Class 1: Good Weld, Class 2: Defect
        defective_dets = [
            d for d in detections
            if d.get("class_id") in (0, 2) or any(k in d["class_name"].lower() for k in ("bad", "defect", "crack", "porosity"))
        ]
        good_dets = [
            d for d in detections
            if d.get("class_id") == 1 or "good" in d["class_name"].lower()
        ]

        if defective_dets:
            top_defect = max(defective_dets, key=lambda d: d["confidence"])
            return {
                "status": "defective",
                "label": f"DEFECT DETECTED: {top_defect['class_name']}",
                "color": "#ef4444",
                "badge": "danger",
                "confidence": top_defect["confidence_pct"],
                "summary": f"Identified {len(defective_dets)} defect anomaly/anomalies (Top: {top_defect['class_name']} at {top_defect['confidence_pct']})."
            }
        elif good_dets:
            top_good = max(good_dets, key=lambda d: d["confidence"])
            return {
                "status": "good",
                "label": "GOOD WELD: Passed Inspection",
                "color": "#22c55e",
                "badge": "success",
                "confidence": top_good["confidence_pct"],
                "summary": f"Weld seam verified acceptable ({top_good['confidence_pct']} confidence)."
            }
        else:
            top_obj = max(detections, key=lambda d: d["confidence"])
            return {
                "status": "detected",
                "label": f"{len(detections)} Detections ({top_obj['class_name']})",
                "color": "#38bdf8",
                "badge": "info",
                "confidence": top_obj["confidence_pct"],
                "summary": f"Detected {len(detections)} object(s)."
            }
