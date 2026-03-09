"""
Clients Package
Harici servis istemcileri

Bu paket şunları içerir:
- yolo_client: YOLO model istemcisi
- huggingface_client: HuggingFace modelleri (SAM, DETR, YOLOS)
"""

from clients import yolo_client, huggingface_client

# Yeni HuggingFace Vision sınıflarını export et
from clients.huggingface_client import (
    HuggingFaceClient,
    HuggingFaceInferenceClient,
    HuggingFaceSpacesClient,
    HuggingFaceVisionClient,
    SAMModel,
    DetectionModel,
    VisionResult
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
    "VisionResult"
]
