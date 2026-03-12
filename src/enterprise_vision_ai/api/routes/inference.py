"""
API Routes - Inference Endpoints
Defect Detection & Ore Classification API Endpoints
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.dependencies import get_inference_service, get_model_service
from api.schemas.request import InferenceRequest, ModelListRequest
from api.schemas.response import (
    ClassificationResult,
    DetectionResult,
    ErrorResponse,
    InferenceResponse,
    ModelInfo,
)
from services.inference_service import InferenceService
from services.model_service import ModelService

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/detect/defects",
    response_model=InferenceResponse,
    summary="Defect Detection",
    description="Detect defects in uploaded images using YOLO model",
)
async def detect_defects(
    body: InferenceRequest = Body(...),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Defekt tespiti endpoint'i.

    Görüntülerdeki defektleri (çatlak, çizik, delik, leke, deformasyon) tespit eder.

    - **image**: Base64 encoded image or image URL
    - **confidence_threshold**: Minimum confidence threshold (0-1)
    - **model_name**: Model name to use for inference
    """
    try:
        start_time = time.time()

        # Inference service ile defekt tespiti yap
        result = await inference_service.detect_defects(
            image_data=body.image,
            confidence_threshold=body.confidence_threshold,
            model_name=body.model_name,
        )

        processing_time = time.time() - start_time

        return InferenceResponse(
            success=True,
            task_type="defect_detection",
            results=result.get("detections", []),
            processing_time=processing_time,
            model_used=result.get("model_used", "unknown"),
            image_info=result.get("image_info", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Defekt tespiti sırasında hata oluştu: {str(e)}",
        )


@router.post(
    "/detect/defects/file",
    response_model=InferenceResponse,
    summary="Defect Detection (File Upload)",
    description="Detect defects using file upload",
)
@limiter.limit("30/minute")
async def detect_defects_file(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = File(0.25, ge=0.0, le=1.0),
    model_name: Optional[str] = File(None),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Defekt tespiti - Dosya yükleme ile.

    Dosya yükleme yöntemiyle defekt tespiti yapar.
    """
    try:
        # Dosya tipini kontrol et
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Desteklenmeyen dosya tipi. JPEG, PNG veya WebP kullanın.",
            )

        start_time = time.time()

        # Dosyayı oku
        contents = await file.read()

        # Inference service ile defekt tespiti yap
        result = await inference_service.detect_defects_from_bytes(
            image_bytes=contents, confidence_threshold=confidence_threshold, model_name=model_name
        )

        processing_time = time.time() - start_time

        return InferenceResponse(
            success=True,
            task_type="defect_detection",
            results=result.get("detections", []),
            processing_time=processing_time,
            model_used=result.get("model_used", "unknown"),
            image_info=result.get("image_info", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Defekt tespiti sırasında hata oluştu: {str(e)}",
        )


@router.post(
    "/classify/ore",
    response_model=InferenceResponse,
    summary="Ore Classification",
    description="Classify ore types in uploaded images",
)
async def classify_ore(
    body: InferenceRequest = Body(...),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Cevher sınıflandırma endpoint'i.

    Görüntülerdeki cevher türlerini (manyetit, krom, atık, düşük tenör) sınıflandırır.

    - **image**: Base64 encoded image or image URL
    - **confidence_threshold**: Minimum confidence threshold (0-1)
    - **model_name**: Model name to use for inference
    """
    try:
        start_time = time.time()

        # Inference service ile cevher sınıflandırması yap
        result = await inference_service.classify_ore(
            image_data=body.image,
            confidence_threshold=body.confidence_threshold,
            model_name=body.model_name,
        )

        processing_time = time.time() - start_time

        return InferenceResponse(
            success=True,
            task_type="ore_classification",
            results=result.get("detections", []),
            processing_time=processing_time,
            model_used=result.get("model_used", "unknown"),
            image_info=result.get("image_info", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cevher sınıflandırma sırasında hata oluştu: {str(e)}",
        )


@router.post(
    "/classify/ore/file",
    response_model=InferenceResponse,
    summary="Ore Classification (File Upload)",
    description="Classify ore types using file upload",
)
@limiter.limit("30/minute")
async def classify_ore_file(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = File(0.25, ge=0.0, le=1.0),
    model_name: Optional[str] = File(None),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Cevher sınıflandırma - Dosya yükleme ile.

    Dosya yükleme yöntemiyle cevher sınıflandırması yapar.
    """
    try:
        # Dosya tipini kontrol et
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Desteklenmeyen dosya tipi. JPEG, PNG veya WebP kullanın.",
            )

        start_time = time.time()

        # Dosyayı oku
        contents = await file.read()

        # Inference service ile cevher sınıflandırması yap
        result = await inference_service.classify_ore_from_bytes(
            image_bytes=contents, confidence_threshold=confidence_threshold, model_name=model_name
        )

        processing_time = time.time() - start_time

        return InferenceResponse(
            success=True,
            task_type="ore_classification",
            results=result.get("detections", []),
            processing_time=processing_time,
            model_used=result.get("model_used", "unknown"),
            image_info=result.get("image_info", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cevher sınıflandırma sırasında hata oluştu: {str(e)}",
        )


@router.get(
    "/models",
    response_model=List[ModelInfo],
    summary="List Available Models",
    description="Get list of available models for inference",
)
@limiter.limit("60/minute")
async def list_models(request: Request, model_service: ModelService = Depends(get_model_service)):
    """
    Mevcut modelleri listele.

    Kullanılabilir tüm YOLO modellerini döndürür.
    """
    try:
        models = await model_service.list_models()
        return models
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model listesi alınırken hata oluştu: {str(e)}",
        )


@router.post("/models/load", summary="Load Model", description="Load a specific model into memory")
@limiter.limit("10/minute")
async def load_model(
    request: Request, model_name: str, model_service: ModelService = Depends(get_model_service)
):
    """
    Model yükle.

    Belirtilen modeli belleğe yükler.
    """
    try:
        success = await model_service.load_model(model_name)
        if success:
            return {"success": True, "message": f"{model_name} modeli yüklendi."}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"{model_name} modeli yüklenemedi."
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model yüklenirken hata oluştu: {str(e)}",
        )


@router.post(
    "/models/unload", summary="Unload Model", description="Unload a specific model from memory"
)
@limiter.limit("10/minute")
async def unload_model(
    request: Request, model_name: str, model_service: ModelService = Depends(get_model_service)
):
    """
    Modeli bellekten kaldır.

    Belirtilen modeli bellekten kaldırır.
    """
    try:
        success = await model_service.unload_model(model_name)
        if success:
            return {"success": True, "message": f"{model_name} modeli bellekten kaldırıldı."}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{model_name} modeli bellekten kaldırılamadı.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model kaldırılırken hata oluştu: {str(e)}",
        )
