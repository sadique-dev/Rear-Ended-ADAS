"""Object detection module — YOLO-based vehicle detection."""

from src.detection.yolo_detector import COCO_VEHICLE_NAMES, Detection, YoloDetector

__all__ = ["COCO_VEHICLE_NAMES", "Detection", "YoloDetector"]
