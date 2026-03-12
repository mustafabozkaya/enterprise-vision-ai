"""
Model management routes for Enterprise Vision AI API.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from enterprise_vision_ai.models import DefectDetector, OreClassifier

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    version: str
    type: str
    description: str
    classes: List[str]
    input_size: tuple
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)


class ModelListResponse(BaseModel):
    """Model list response."""

    models: List[ModelInfo]
    total: int


class ModelPredictRequest(BaseModel):
    """Model prediction request."""

    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="Image URL")
    confidence: float = Field(0.25, ge=0.0, le=1.0)


class ModelPredictResponse(BaseModel):
    """Model prediction response."""

    model_name: str
    predictions: List[Dict]
    inference_time: float
    timestamp: str


# Available models
AVAILABLE_MODELS = [
    ModelInfo(
        name="defect-detector",
        version="1.0.0",
        type="segmentation",
        description="Surface defect detection model",
        classes=["çatlak", "çizik", "delik", "leke", "deformasyon"],
        input_size=(640, 640),
        confidence_threshold=0.25,
    ),
    ModelInfo(
        name="ore-classifier",
        version="1.0.0",
        type="classification",
        description="Ore classification model",
        classes=["manyetit", "krom", "atık", "düşük tenör"],
        input_size=(640, 640),
        confidence_threshold=0.25,
    ),
]


@router.get(
    "",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List models",
    description="Get list of available models",
)
async def list_models() -> ModelListResponse:
    """
    List all available models.

    Returns:
        ModelListResponse: List of models
    """
    return ModelListResponse(models=AVAILABLE_MODELS, total=len(AVAILABLE_MODELS))


@router.get(
    "/{model_name}",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get model info",
    description="Get detailed information about a specific model",
)
async def get_model(model_name: str) -> ModelInfo:
    """
    Get model information.

    Args:
        model_name: Model name

    Returns:
        ModelInfo: Model information

    Raises:
        HTTPException: If model not found
    """
    for model in AVAILABLE_MODELS:
        if model.name == model_name:
            return model

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model '{model_name}' not found",
    )


@router.post(
    "/{model_name}/predict",
    response_model=ModelPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Run prediction",
    description="Run inference with specified model",
)
async def predict(model_name: str, request: ModelPredictRequest) -> ModelPredictResponse:
    """
    Run model prediction.

    Args:
        model_name: Model name
        request: Prediction request

    Returns:
        ModelPredictResponse: Prediction results

    Raises:
        HTTPException: If model not found or prediction fails
    """
    from datetime import datetime

    # TODO: Implement actual prediction logic
    # For now, return mock response
    if model_name not in ["defect-detector", "ore-classifier"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    return ModelPredictResponse(
        model_name=model_name,
        predictions=[],
        inference_time=0.0,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get(
    "/{model_name}/versions",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="List model versions",
    description="Get available versions for a model",
)
async def list_model_versions(model_name: str) -> List[str]:
    """
    List model versions.

    Args:
        model_name: Model name

    Returns:
        List[str]: Available versions

    Raises:
        HTTPException: If model not found
    """
    if model_name not in ["defect-detector", "ore-classifier"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    return ["1.0.0", "1.0.0-beta"]
