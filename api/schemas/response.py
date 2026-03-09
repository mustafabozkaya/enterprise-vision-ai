"""
API Schemas - Response Models
Pydantic modelleri - Yanıt yapıları
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box koordinatları"""
    x1: float = Field(..., description="Sol üst x koordinatı")
    y1: float = Field(..., description="Sol üst y koordinatı")
    x2: float = Field(..., description="Sağ alt x koordinatı")
    y2: float = Field(..., description="Sağ alt y koordinatı")
    
    @property
    def width(self) -> float:
        """Genişlik"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Yükseklik"""
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple:
        """Merkez nokta"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class DetectionResult(BaseModel):
    """
    Tek bir tespit sonucu.
    """
    class_name: str = Field(
        ...,
        description="Tespit edilen sınıfın adı (Türkçe)",
        examples=["manyetit", "krom", "çatlak", "çizik"]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Güven skoru"
    )
    bbox: BoundingBox = Field(
        ...,
        description="Bounding box koordinatları"
    )
    class_id: int = Field(
        ...,
        description="Sınıf ID'si"
    )
    mask: Optional[List[List[int]]] = Field(
        default=None,
        description="Segmentasyon maskesi (polygon koordinatları)"
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity seviyesi (düşük, orta, yüksek)",
        examples=["düşük", "orta", "yüksek"]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Ek meta veriler"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "class_name": "manyetit",
                "confidence": 0.95,
                "bbox": {"x1": 100.0, "y1": 50.0, "x2": 300.0, "y2": 250.0},
                "class_id": 0,
                "severity": None,
                "metadata": {"ore_type": "magnetite"}
            }
        }


class ClassificationResult(BaseModel):
    """
    Sınıflandırma sonucu.
    """
    class_name: str = Field(
        ...,
        description="Sınıflandırılan sınıfın adı (Türkçe)",
        examples=["manyetit", "krom", "atık"]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Güven skoru"
    )
    class_id: int = Field(
        ...,
        description="Sınıf ID'si"
    )
    probability: Optional[float] = Field(
        default=None,
        description="Olasılık değeri"
    )


class ImageInfo(BaseModel):
    """
    Görüntü bilgileri.
    """
    width: int = Field(..., description="Görüntü genişliği")
    height: int = Field(..., description="Görüntü yüksekliği")
    channels: Optional[int] = Field(
        default=3,
        description="Kanal sayısı"
    )
    format: Optional[str] = Field(
        default=None,
        description="Görüntü formatı"
    )
    size_bytes: Optional[int] = Field(
        default=None,
        description="Dosya boyutu (bytes)"
    )


class Metrics(BaseModel):
    """
    İşlem metrikleri.
    """
    total_detections: int = Field(
        default=0,
        description="Toplam tespit sayısı"
    )
    processing_time: float = Field(
        ...,
        description="İşleme süresi (saniye)"
    )
    inference_time: Optional[float] = Field(
        default=None,
        description="Inference süresi (saniye)"
    )
    metal_ratio: Optional[float] = Field(
        default=None,
        description="Metal oranı (%)"
    )
    anomaly_score: Optional[float] = Field(
        default=None,
        description="Anomali skoru (0-100)"
    )
    severity_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Severity dağılımı"
    )
    class_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Sınıf dağılımı"
    )


class InferenceResponse(BaseModel):
    """
    Inference yanıtı.
    """
    success: bool = Field(
        ...,
        description="İşlem başarılı mı?"
    )
    task_type: str = Field(
        ...,
        description="Görev türü",
        examples=["defect_detection", "ore_classification"]
    )
    results: List[DetectionResult] = Field(
        default_factory=list,
        description="Tespit sonuçları"
    )
    processing_time: float = Field(
        ...,
        description="Toplam işleme süresi (saniye)"
    )
    model_used: str = Field(
        ...,
        description="Kullanılan model"
    )
    image_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Görüntü bilgileri"
    )
    metrics: Optional[Metrics] = Field(
        default=None,
        description="Metrikler"
    )
    message: Optional[str] = Field(
        default=None,
        description="Ek mesaj"
    )
    error: Optional[str] = Field(
        default=None,
        description="Hata mesajı (başarısız ise)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "task_type": "defect_detection",
                "results": [
                    {
                        "class_name": "çatlak",
                        "confidence": 0.92,
                        "bbox": {"x1": 100.0, "y1": 50.0, "x2": 300.0, "y2": 250.0},
                        "class_id": 0,
                        "severity": "yüksek"
                    }
                ],
                "processing_time": 0.156,
                "model_used": "yolo11n-seg.pt",
                "image_info": {"width": 640, "height": 640, "channels": 3}
            }
        }


class UploadedFile(BaseModel):
    """
    Yüklenen dosya bilgileri.
    """
    filename: str = Field(..., description="Orijinal dosya adı")
    stored_path: str = Field(..., description="Kaydedilen dosya yolu")
    size: int = Field(..., description="Dosya boyutu (bytes)")
    content_type: str = Field(..., description="Dosya içerik tipi")


class UploadResponse(BaseModel):
    """
    Dosya yükleme yanıtı.
    """
    success: bool = Field(..., description="Yükleme başarılı mı?")
    file: UploadedFile = Field(..., description="Yüklenen dosya bilgileri")
    message: str = Field(..., description="Mesaj")
    processing_time: float = Field(..., description="İşleme süresi")


class BatchUploadResponse(BaseModel):
    """
    Toplu yükleme yanıtı.
    """
    success: bool = Field(..., description="Yükleme başarılı mı?")
    files: List[UploadedFile] = Field(..., description="Yüklenen dosyalar")
    total_count: int = Field(..., description="Toplam dosya sayısı")
    success_count: int = Field(..., description="Başarılı yükleme sayısı")
    error_count: int = Field(..., description="Hatalı yükleme sayısı")
    errors: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Hata listesi"
    )
    message: str = Field(..., description="Mesaj")
    processing_time: float = Field(..., description="İşleme süresi")


class ModelInfo(BaseModel):
    """
    Model bilgileri.
    """
    name: str = Field(..., description="Model adı")
    type: str = Field(..., description="Model türü")
    path: Optional[str] = Field(default=None, description="Model dosya yolu")
    loaded: bool = Field(..., description="Bellekte yüklü mü?")
    size_mb: Optional[float] = Field(default=None, description="Model boyutu (MB)")
    task_type: str = Field(..., description="Görev türü")
    classes: Optional[List[str]] = Field(
        default=None,
        description="Sınıf listesi"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Ek meta veriler"
    )


class HealthResponse(BaseModel):
    """
    Sağlık kontrolü yanıtı.
    """
    status: str = Field(..., description="API durumu", examples=["healthy", "degraded", "unhealthy"])
    version: str = Field(..., description="API versiyonu")
    models_loaded: List[str] = Field(
        default_factory=list,
        description="Yüklenmiş modeller"
    )
    models_count: int = Field(..., description="Yüklenmiş model sayısı")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Zaman damgası"
    )
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="Çalışma süresi (saniye)"
    )


class ErrorResponse(BaseModel):
    """
    Hata yanıtı.
    """
    error: str = Field(..., description="Hata türü")
    detail: str = Field(..., description="Hata detayı")
    code: Optional[str] = Field(default=None, description="Hata kodu")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Zaman damgası"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="İstek ID'si"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "confidence_threshold değeri 0 ile 1 arasında olmalıdır",
                "code": "VALIDATION_ERROR",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class APIInfo(BaseModel):
    """
    API bilgileri.
    """
    name: str = Field(..., description="API adı")
    version: str = Field(..., description="API versiyonu")
    description: str = Field(..., description="API açıklaması")
    docs_url: str = Field(..., description="Dokümantasyon URL'si")
    endpoints: Dict[str, Any] = Field(..., description="Endpoint'ler")
