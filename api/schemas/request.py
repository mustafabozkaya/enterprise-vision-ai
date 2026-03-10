"""
API Schemas - Request Models
Pydantic modelleri - İstek yapıları
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class TaskType(str, Enum):
    """Görev türleri"""

    DEFECT_DETECTION = "defect_detection"
    ORE_CLASSIFICATION = "ore_classification"


class ImageSource(str, Enum):
    """Görüntü kaynağı türleri"""

    BASE64 = "base64"
    URL = "url"
    FILE_PATH = "file_path"


class InferenceRequest(BaseModel):
    """
    Inference isteği için temel model.

    Görüntü analizi için kullanılan istek yapısı.
    """

    image: str = Field(
        ...,
        description="Görüntü (base64 encoded, URL veya dosya yolu)",
        examples=[
            "base64_encoded_image_string",
            "https://example.com/image.jpg",
            "/path/to/image.jpg",
        ],
    )
    image_source: ImageSource = Field(
        default=ImageSource.BASE64, description="Görüntü kaynağı türü"
    )
    confidence_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Minimum güven eşiği"
    )
    model_name: Optional[str] = Field(default=None, description="Kullanılacak model adı")
    iou_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0, description="IoU eşiği (Non-Maximum Suppression)"
    )
    max_detections: int = Field(default=100, ge=1, le=300, description="Maksimum tespit sayısı")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Görüntü değerini doğrular"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Görüntü değeri boş olamaz")
        return v.strip()


class DefectDetectionRequest(InferenceRequest):
    """
    Defekt tespiti isteği.

    Görüntülerdeki defektleri tespit etmek için kullanılır.
    """

    task_type: TaskType = Field(default=TaskType.DEFECT_DETECTION, description="Görev türü")
    include_severity: bool = Field(default=True, description="Severity seviyesini dahil et")
    defect_classes: Optional[List[str]] = Field(
        default=None,
        description="Tespit edilecek defekt türleri (boş ise tümü)",
        examples=[["çatlak", "çizik"], ["delik"]],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image": "base64_encoded_image...",
                "image_source": "base64",
                "confidence_threshold": 0.25,
                "model_name": "yolo11n-seg.pt",
                "include_severity": True,
                "defect_classes": ["çatlak", "çizik", "delik", "leke", "deformasyon"],
            }
        }


class OreClassificationRequest(InferenceRequest):
    """
    Cevher sınıflandırma isteği.

    Görüntülerdeki cevher türlerini sınıflandırmak için kullanılır.
    """

    task_type: TaskType = Field(default=TaskType.ORE_CLASSIFICATION, description="Görev türü")
    ore_classes: Optional[List[str]] = Field(
        default=None,
        description="Sınıflandırılacak cevher türleri (boş ise tümü)",
        examples=[["manyetit", "krom"], ["atık"]],
    )
    return_metrics: bool = Field(default=True, description="Metrikleri döndür")

    class Config:
        json_schema_extra = {
            "example": {
                "image": "base64_encoded_image...",
                "image_source": "base64",
                "confidence_threshold": 0.25,
                "model_name": "yolo11n-seg.pt",
                "ore_classes": ["manyetit", "krom", "atık", "düşük tenör"],
                "return_metrics": True,
            }
        }


class UploadRequest(BaseModel):
    """
    Dosya yükleme isteği.
    """

    subdir: str = Field(default="images", description="Kaydedilecek alt dizin")
    max_file_size: int = Field(
        default=10 * 1024 * 1024, description="Maksimum dosya boyutu (bytes)"
    )


class BatchUploadRequest(BaseModel):
    """
    Toplu yükleme isteği.
    """

    subdir: str = Field(default="batch", description="Kaydedilecek alt dizin")
    max_files: int = Field(default=20, ge=1, le=50, description="Maksimum dosya sayısı")
    wait_for_processing: bool = Field(default=False, description="Yükleme sonrası işleme bekle")


class ModelListRequest(BaseModel):
    """
    Model listesi isteği.
    """

    include_loaded_only: bool = Field(
        default=False, description="Sadece yüklenmiş modelleri dahil et"
    )
    model_type: Optional[str] = Field(default=None, description="Model türü filtresi")


class HealthCheckRequest(BaseModel):
    """
    Sağlık kontrolü isteği.
    """

    include_models: bool = Field(default=True, description="Model durumlarını dahil et")
    include_system: bool = Field(default=True, description="Sistem bilgilerini dahil et")
