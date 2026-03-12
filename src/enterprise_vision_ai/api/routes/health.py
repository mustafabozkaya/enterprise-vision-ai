"""
Health check routes for Enterprise Vision AI API.
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


class ReadyResponse(BaseModel):
    """Readiness check response model."""

    ready: bool
    checks: Dict[str, bool]


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check API health status",
)
async def health_check() -> HealthResponse:
    """
    Get API health status.

    Returns:
        HealthResponse: Health status information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0.0",
        services={
            "api": "up",
            "inference": "up",
            "storage": "up",
        },
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if API is ready to accept requests",
)
async def readiness_check() -> ReadyResponse:
    """
    Check API readiness.

    Returns:
        ReadyResponse: Readiness status
    """
    # TODO: Add actual readiness checks (DB connection, model loading, etc.)
    return ReadyResponse(
        ready=True,
        checks={
            "database": True,
            "model_service": True,
            "storage": True,
        },
    )


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if API is alive",
)
async def liveness_check() -> Dict[str, str]:
    """
    Check API liveness.

    Returns:
        dict: Liveness status
    """
    return {"status": "alive"}
