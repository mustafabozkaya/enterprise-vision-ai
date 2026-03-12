"""
API Dependencies - FastAPI Bağımlılıkları
Model yükleme, authentication ve rate limiting için bağımlılıklar
"""

from functools import lru_cache
from typing import Any, Dict, Optional

from clients.yolo_client import YOLOModelManager
from fastapi import Depends, Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from services.inference_service import InferenceService
from services.model_service import ModelService

# =============================================================================
# Singleton Instances
# =============================================================================


@lru_cache()
def get_limiter() -> Limiter:
    """Rate limiter instance'ı döndürür"""
    return Limiter(key_func=get_remote_address)


@lru_cache()
def get_yolo_manager() -> YOLOModelManager:
    """YOLO model manager instance'ı döndürür"""
    return YOLOModelManager()


def get_inference_service() -> InferenceService:
    """
    Inference service instance'ı döndürür.

    Returns:
        InferenceService: Inference servis instance'ı
    """
    manager = get_yolo_manager()
    return InferenceService(manager)


def get_model_service() -> ModelService:
    """
    Model service instance'ı döndürür.

    Returns:
        ModelService: Model servis instance'ı
    """
    manager = get_yolo_manager()
    return ModelService(manager)


# =============================================================================
# Authentication Dependencies
# =============================================================================


async def verify_api_key(x_api_key: Optional[str] = Header(None, description="API Key")) -> str:
    """
    API key doğrulama.

    Header'dan API key'i alır ve doğrular.
    """
    # Development modunda API key kontrolünü atla
    # Production'da gerçek API key kontrolü yapılmalı
    if x_api_key is None:
        # API key zorunlu değilse geçici bir değer kullan
        return "development"

    return x_api_key


async def verify_admin_key(
    x_admin_key: Optional[str] = Header(None, description="Admin API Key")
) -> str:
    """
    Admin API key doğrulama.

    Admin işlemleri için admin key'i doğrular.
    """
    if x_admin_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin API key gerekli"
        )

    # Gerçek bir admin key kontrolü yapılmalı
    # Şimdilik basit bir kontrol
    if len(x_admin_key) < 8:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Geçersiz admin API key"
        )

    return x_admin_key


# =============================================================================
# Rate Limiting Dependencies
# =============================================================================


def get_rate_limit_key(request) -> str:
    """
    Rate limiting için unique key oluşturur.

    Args:
        request: FastAPI request object

    Returns:
        str: Unique key
    """
    # IP adresi veya API key kullan
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0]
    else:
        ip = request.client.host

    return f"{ip}:{request.url.path}"


# =============================================================================
# Model Dependencies
# =============================================================================


async def get_model_for_inference(
    model_name: Optional[str] = None,
    task_type: str = "detection",
    model_service: ModelService = Depends(get_model_service),
) -> str:
    """
    Inference için model adı döndürür.

    Args:
        model_name: İstenen model adı
        task_type: Görev türü
        model_service: Model servis instance'ı

    Returns:
        str: Model adı

    Raises:
        HTTPException: Model bulunamazsa
    """
    if model_name is None:
        # Varsayılan model
        if task_type == "segmentation":
            model_name = "yolo11n-seg.pt"
        else:
            model_name = "yolo11n.pt"

    # Modelin yüklü olup olmadığını kontrol et
    loaded_models = model_service.get_loaded_models()
    if model_name not in loaded_models:
        # Modeli yüklemeye çalış
        success = await model_service.load_model(model_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' bulunamadı veya yüklenemedi",
            )

    return model_name


# =============================================================================
# Request Validation Dependencies
# =============================================================================


def validate_content_type(content_type: Optional[str] = Header(None)) -> str:
    """
    Content type doğrulama.

    Args:
        content_type: İçerik tipi

    Returns:
        str: Doğrulanmış content type

    Raises:
        HTTPException: Desteklenmeyen content type
    """
    allowed_types = ["application/json", "multipart/form-data", "application/x-www-form-urlencoded"]

    if content_type and content_type not in allowed_types:
        # Partial match kontrolü
        if not any(content_type.startswith(ct) for ct in allowed_types):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Desteklenmeyen içerik tipi: {content_type}",
            )

    return content_type or "application/json"


def validate_accept_header(accept: Optional[str] = Header(None)) -> str:
    """
    Accept header doğrulama.

    Args:
        accept: Accept header değeri

    Returns:
        str: Doğrulanmış accept değeri
    """
    allowed = ["application/json", "*/*"]

    if accept and accept not in allowed:
        if not any(accept.startswith(a) for a in allowed):
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail=f"Desteklenmeyen accept header: {accept}",
            )

    return accept or "application/json"


# =============================================================================
# Utility Dependencies
# =============================================================================


async def get_request_id(
    x_request_id: Optional[str] = Header(None, description="Request ID")
) -> str:
    """
    Request ID döndürür veya oluşturur.

    Args:
        x_request_id: Header'dan gelen request ID

    Returns:
        str: Request ID
    """
    import uuid

    return x_request_id or str(uuid.uuid4())


class PaginationParams:
    """Sayfalama parametreleri"""

    def __init__(self, page: int = 1, page_size: int = 20, max_page_size: int = 100):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sayfa numarası 1'den büyük olmalıdır",
            )
        if page_size < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Sayfa boyutu 1'den büyük olmalıdır"
            )
        if page_size > max_page_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sayfa boyutu {max_page_size}'i geçemez",
            )

        self.page = page
        self.page_size = page_size
        self.skip = (page - 1) * page_size


def get_pagination_params(page: int = 1, page_size: int = 20) -> PaginationParams:
    """
    Sayfalama parametreleri döndürür.

    Args:
        page: Sayfa numarası
        page_size: Sayfa boyutu

    Returns:
        PaginationParams: Sayfalama parametreleri
    """
    return PaginationParams(page=page, page_size=page_size)
