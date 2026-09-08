"""
detector.py
-----------
Purpose:
    Wraps the Ultralytics YOLOv8 model and exposes an advanced, production-grade
    inference interface for road-damage detection on images, videos, and real-time
    webcam streams.

    Features:
    - Intelligent class mapping (maps '0' to 'Pothole' / 'Road Damage').
    - Engineering severity scoring (Critical, High, Moderate, Minor).
    - Pavement Condition Index (PCI) / Road Health Index (0-100).
    - Actionable maintenance recommendation & asphalt material estimation.
    - Sleek HUD visual annotation with corner-bracket viewfinders and glowing
      severity indicators.
    - High-speed single frame inference for live webcam/dashcam streaming.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config
from logger import get_logger

logger = get_logger(__name__)

# BGR color palette for HUD annotations
COLOR_PALETTE = {
    "Critical": (50, 50, 245),      # Crimson Red (#ef4444)
    "High": (25, 115, 245),          # Hazard Orange (#f97316)
    "Moderate": (25, 190, 245),      # Amber Yellow (#f59e0b)
    "Minor": (220, 210, 45),         # Cyan (#06b6d4)
    "Clear": (80, 200, 80),          # Emerald (#10b981)
}


class RoadDamageDetector:
    """Singleton-style wrapper around a YOLOv8 model with engineering analytics."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.model_path = None
        self.class_names: List[str] = []
        self.using_custom_model = False
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """Load YOLO model, preferring custom best.pt over yolov8n.pt."""
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = None

        if config.CUSTOM_MODEL_PATH.exists():
            file_size = config.CUSTOM_MODEL_PATH.stat().st_size
            if file_size < config.MODEL_MIN_SIZE_BYTES:
                logger.warning(
                    "best.pt exists but is only %d bytes — NOT a valid model. "
                    "Falling back to yolov8n.pt.", file_size,
                )
            else:
                model_path = config.CUSTOM_MODEL_PATH
                self.using_custom_model = True
                logger.info("Custom model found and validated. Loading best.pt")

        if model_path is None:
            model_path = config.FALLBACK_MODEL_PATH
            logger.warning(
                "Using fallback yolov8n.pt (COCO model — will NOT detect road damage). "
                "Train/download a road-damage model as best.pt for real detection."
            )

        try:
            self.model = YOLO(str(model_path))
            self.model_path = model_path
            self.class_names = list(self.model.names.values())
            logger.info("YOLO model loaded from %s | raw classes=%s", model_path, self.class_names)
        except Exception as exc:
            logger.exception("Failed to load YOLO model: %s", exc)
            raise RuntimeError(
                f"Could not load YOLO model from {model_path}. "
                "Ensure ultralytics is installed and the model file is valid."
            ) from exc

    # ------------------------------------------------------------------ CLASS & SEVERITY LOGIC

    def _resolve_class_name(self, class_id: int) -> str:
        """Translate raw model class index to human-readable road damage title."""
        raw_name = str(class_id)
        if class_id < len(self.class_names):
            raw_name = str(self.class_names[class_id]).strip()

        # Check explicit overrides in config
        if raw_name in config.CLASS_NAME_OVERRIDES:
            return config.CLASS_NAME_OVERRIDES[raw_name]

        # Clean fallback
        if raw_name.lower() in ("0", "damage", "defect"):
            return "Pothole"

        return raw_name.replace("_", " ").title()

    @staticmethod
    def _calculate_severity(area_pct: float, conf: float) -> str:
        """Compute civil engineering defect severity rating."""
        if area_pct >= 3.0 or (area_pct >= 2.0 and conf >= 0.82):
            return "Critical"
        if area_pct >= 1.2 or (area_pct >= 0.8 and conf >= 0.75):
            return "High"
        if area_pct >= 0.35 or conf >= 0.50:
            return "Moderate"
        return "Minor"

    @staticmethod
    def _get_repair_action(class_name: str, severity: str, area_pct: float) -> str:
        """Generate maintenance prescription and material estimate."""
        if severity == "Critical":
            return f"Emergency cold-mix asphalt patch (~{max(15, int(area_pct * 8))} kg) required immediately. Potential tire/suspension hazard."
        if severity == "High":
            return f"High priority repair: Mill-and-fill or hot-pour mastic asphalt (~{max(8, int(area_pct * 5))} kg) within 7 days."
        if severity == "Moderate":
            return f"Scheduled maintenance: Polymer crack seal / surface patch (~{max(4, int(area_pct * 3))} kg) during regular road audit."
        return "Routine monitoring: Early-stage surface wear. Re-survey in next quarterly inspection."

    @staticmethod
    def _compute_road_health_index(boxes_data: List[Dict[str, Any]]) -> Tuple[int, str, str]:
        """
        Calculate Pavement Condition Index (PCI) / Road Health Index (0-100).
        Returns: (health_index, overall_severity, overall_recommendation)
        """
        if not boxes_data:
            return 100, "Clear", "Pristine Road Condition: No actionable surface defects detected. Road complies with safety standards."

        score = 100
        severity_counts = {"Critical": 0, "High": 0, "Moderate": 0, "Minor": 0}

        for box in boxes_data:
            sev = box.get("severity", "Moderate")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            penalty = config.SEVERITY_PENALTIES.get(sev, 8)
            score -= penalty

        # Floor score at 10 if defects exist
        score = max(score, 10)

        if severity_counts["Critical"] > 0:
            overall_severity = "Critical"
            recommendation = (
                f"CRITICAL SAFETY HAZARD ({severity_counts['Critical']} urgent pothole/void detected). "
                "Immediate road authority intervention required to prevent vehicle loss-of-control."
            )
        elif severity_counts["High"] > 0:
            overall_severity = "High"
            recommendation = (
                f"HIGH PRIORITY DEFECTS ({severity_counts['High']} major defects). "
                "Schedule asphalt patching within 7–14 days to prevent sub-base water damage."
            )
        elif severity_counts["Moderate"] > 0:
            overall_severity = "Moderate"
            recommendation = (
                f"MODERATE ROAD WEAR ({severity_counts['Moderate']} defects logged). "
                "Include in upcoming municipal preventative maintenance cycle."
            )
        else:
            overall_severity = "Minor"
            recommendation = "MINOR SURFACE ABRASION: Surface micro-cracks present. Monitor during next inspection."

        return score, overall_severity, recommendation

    # ------------------------------------------------------------------ HUD VISUAL ANNOTATOR

    def _draw_hud_annotations(self, image: np.ndarray, boxes_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw clean, high-tech engineering HUD annotations on image.
        Features corner-bracket viewfinders, subtle translucent shaded fills,
        and floating telemetry chips with severity colors.
        """
        annotated = image.copy()
        overlay = image.copy()
        h, w = image.shape[:2]

        for item in boxes_data:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            sev = item.get("severity", "Moderate")
            color = COLOR_PALETTE.get(sev, (25, 190, 245))

            # 1. Subtle translucent fill inside bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # 2. Main border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            # 3. Corner bracket "viewfinder" accents (thick corner lines)
            corner_len = max(8, min(24, int(min(x2 - x1, y2 - y1) * 0.25)))
            thickness = 3

            # Top-Left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thickness)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thickness)
            # Top-Right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thickness)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thickness)
            # Bottom-Left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, thickness)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, thickness)
            # Bottom-Right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, thickness)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, thickness)

            # 4. Floating telemetry tag
            label_text = f"{item['class_name']} · {sev.upper()} {int(item['confidence']*100)}%"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.52
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            tag_y1 = max(0, y1 - text_h - 10)
            tag_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 12
            tag_x1 = x1
            tag_x2 = min(w - 1, x1 + text_w + 14)

            # Tag background
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), (18, 20, 24), -1)
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1)

            # Tag indicator dot
            cv2.circle(annotated, (tag_x1 + 7, tag_y1 + (tag_y2 - tag_y1) // 2), 3, color, -1)

            # Tag text
            cv2.putText(
                annotated,
                label_text,
                (tag_x1 + 14, tag_y2 - 6),
                font,
                font_scale,
                (245, 245, 245),
                font_thickness,
                cv2.LINE_AA,
            )

        # Blend overlay (alpha = 0.12)
        cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
        return annotated

    # ------------------------------------------------------------------ IMAGE DETECTION

    def detect_image(self, image_path: Path, output_path: Path) -> Dict[str, Any]:
        start_time = time.time()

        image_path = Path(image_path).resolve()
        output_path = Path(output_path).resolve()

        try:
            raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        except Exception:
            image_bgr = None

        if image_bgr is None:
            image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise ValueError(f"Could not read image file: {image_path}")
        img_h, img_w = image_bgr.shape[:2]

        results = self.model.predict(
            source=image_bgr,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        boxes_data, damage_types, confidences = self._extract_rich_detections(result, img_w, img_h)

        # Draw our custom engineering HUD annotations
        annotated_frame = self._draw_hud_annotations(image_bgr, boxes_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix or ".jpg"
        success, encoded = cv2.imencode(ext, annotated_frame)
        if success:
            encoded.tofile(str(output_path))
        else:
            cv2.imwrite(str(output_path), annotated_frame)

        processing_time = round(time.time() - start_time, 3)
        return self._build_rich_summary(boxes_data, damage_types, confidences, processing_time, output_path)

    # ------------------------------------------------------------------ VIDEO DETECTION

    def detect_video(self, video_path: Path, output_path: Path) -> Dict[str, Any]:
        start_time = time.time()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        capture.release()

        if duration > config.VIDEO_MAX_DURATION_SECONDS:
            raise ValueError(
                f"Video is {duration:.0f}s long. Maximum allowed is "
                f"{config.VIDEO_MAX_DURATION_SECONDS}s. Trim the video and try again."
            )

        vid_stride = config.VIDEO_VID_STRIDE
        imgsz = config.VIDEO_IMG_SIZE
        half = config.USE_HALF_PRECISION and self._is_gpu_available()

        all_boxes_data: List[Dict[str, Any]] = []
        all_damage_types: List[str] = []
        all_confidences: List[float] = []
        annotated_frames: List[Any] = []

        results_stream = self.model.predict(
            source=str(video_path),
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=imgsz,
            half=half,
            vid_stride=vid_stride,
            stream=True,
            verbose=False,
        )

        for result in results_stream:
            orig_img = result.orig_img
            h, w = orig_img.shape[:2]
            boxes_data, damage_types, confidences = self._extract_rich_detections(result, w, h)
            all_boxes_data.extend(boxes_data)
            all_damage_types.extend(damage_types)
            all_confidences.extend(confidences)

            hud_frame = self._draw_hud_annotations(orig_img, boxes_data)
            annotated_frames.append(hud_frame)

        output_fps = max(fps / vid_stride, 1.0) if vid_stride > 0 else fps
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        if not writer.isOpened():
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

        try:
            for frame in annotated_frames:
                writer.write(frame)
        finally:
            writer.release()

        processing_time = round(time.time() - start_time, 3)
        return self._build_rich_summary(all_boxes_data, all_damage_types, all_confidences, processing_time, output_path)

    # ------------------------------------------------------------------ REAL-TIME WEBCAM INFERENCE

    def detect_frame(self, frame_bgr: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        High-speed single frame inference for webcam/dashcam streaming.
        Returns detection metadata and hazard flags in < 30ms.
        """
        start_time = time.time()
        h, w = frame_bgr.shape[:2]

        results = self.model.predict(
            source=frame_bgr,
            conf=conf_threshold,
            iou=config.IOU_THRESHOLD,
            imgsz=480,
            verbose=False,
        )
        result = results[0]
        boxes_data, damage_types, confidences = self._extract_rich_detections(result, w, h)

        health_index, severity, recommendation = self._compute_road_health_index(boxes_data)
        has_hazard = len(boxes_data) > 0

        processing_time = round(time.time() - start_time, 3)

        return {
            "has_hazard": has_hazard,
            "detection_count": len(boxes_data),
            "severity": severity,
            "road_health_index": health_index,
            "boxes": boxes_data,
            "damage_types": sorted(set(damage_types)) if damage_types else ["Clear"],
            "processing_time": processing_time,
        }

    # ------------------------------------------------------------------ EXTRACTION & SUMMARIES

    def _extract_rich_detections(
        self, result, img_w: int, img_h: int
    ) -> Tuple[List[Dict[str, Any]], List[str], List[float]]:
        boxes_data: List[Dict[str, Any]] = []
        damage_types: List[str] = []
        confidences: List[float] = []

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return boxes_data, damage_types, confidences

        for idx, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self._resolve_class_name(class_id)

            coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = coords
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)

            total_img_area = max(1.0, float(img_w * img_h))
            area_pct = round(((box_w * box_h) / total_img_area) * 100.0, 2)

            severity = self._calculate_severity(area_pct, confidence)
            repair_action = self._get_repair_action(class_name, severity, area_pct)

            boxes_data.append({
                "id": idx + 1,
                "box": coords,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "area_pct": area_pct,
                "severity": severity,
                "repair_action": repair_action,
            })
            damage_types.append(class_name)
            confidences.append(round(confidence, 4))

        return boxes_data, damage_types, confidences

    def _build_rich_summary(
        self,
        boxes_data: List[Dict[str, Any]],
        damage_types: List[str],
        confidences: List[float],
        processing_time: float,
        output_path: Path,
    ) -> Dict[str, Any]:
        unique_damage_types = sorted(set(damage_types)) if damage_types else ["Clear"]
        average_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        health_index, overall_severity, recommendation = self._compute_road_health_index(boxes_data)

        # Severity distribution counts
        severity_breakdown = {"Critical": 0, "High": 0, "Moderate": 0, "Minor": 0}
        for box in boxes_data:
            s = box.get("severity", "Moderate")
            severity_breakdown[s] = severity_breakdown.get(s, 0) + 1

        return {
            "damage_types": unique_damage_types,
            "detection_count": len(boxes_data),
            "confidences": confidences,
            "average_confidence": average_confidence,
            "processing_time": processing_time,
            "output_path": str(output_path),
            "severity": overall_severity,
            "road_health_index": health_index,
            "repair_recommendation": recommendation,
            "severity_breakdown": severity_breakdown,
            "detection_boxes": boxes_data,
        }

    @staticmethod
    def _is_gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def get_detector() -> RoadDamageDetector:
    return RoadDamageDetector()
