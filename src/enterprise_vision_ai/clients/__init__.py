"""
Clients Package
Harici servis istemcileri

Bu paket şunları içerir:
- yolo_client: YOLO model istemcisi
- huggingface_client: HuggingFace modelleri (SAM, DETR, YOLOS)
"""

from enterprise_vision_ai.clients import huggingface_client, yolo_client

# Yeni HuggingFace Vision sınıflarını export et
from enterprise_vision_ai.clients.huggingface_client import (
    DetectionModel,
    HuggingFaceClient,
    HuggingFaceInferenceClient,
    HuggingFaceSpacesClient,
    HuggingFaceVisionClient,
    SAMModel,
    VisionResult,
)

__all__ = [
    "yolo_client",
    "huggingface_client",
    "HuggingFaceClient",
    "HuggingFaceInferenceClient",
    "HuggingFaceSpacesClient",
    "HuggingFaceVisionClient",
    "SAMModel",
    "DetectionModel",
    "VisionResult",
]
