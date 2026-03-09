"""
API Gateway - FastAPI Uygulaması Giriş Noktası
Computer Vision Projesi için REST API
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.routes import inference, upload
from api.schemas.response import HealthResponse


# Rate limiter yapılandırması
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama yaşam döngüsü yöneticisi.
    Başlangıç ve kapatma işlemleri için kullanılır.
    """
    # Başlangıç: Model yükleme, bağlantı kurma vb.
    print("API Gateway starting...")
    start_time = time.time()
    
    yield
    
    # Kapatma: Kaynakları temizle
    print("API Gateway shutting down...")
    from clients.yolo_client import yolo_model_manager
    yolo_model_manager.unload_all_models()


def create_application() -> FastAPI:
    """
    FastAPI uygulaması oluşturur.
    
    Returns:
        FastAPI: Yapılandırılmış FastAPI uygulaması
    """
    app = FastAPI(
        title="Enterprise Vision AI API",
        description="Computer Vision API - Defect Detection & Ore Classification",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Üretimde belirli domain'lerle değiştirilmeli
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip Sıkıştırma
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Rate Limiting
    app.state.limiter = limiter
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "detail": "Çok fazla istek gönderildi. Lütfen daha sonra tekrar deneyin.",
                "retry_after": exc.detail
            }
        )
    
    # Router'ları ekle
    app.include_router(
        inference.router,
        prefix="/api/v1",
        tags=["Inference"]
    )
    app.include_router(
        upload.router,
        prefix="/api/v1",
        tags=["Upload"]
    )
    
    return app


app = create_application()


@app.get("/", tags=["Root"])
async def root():
    """
    Kök endpoint - API bilgileri döndürür.
    """
    return {
        "name": "Enterprise Vision AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("100/minute")
async def health_check(request: Request):
    """
    Sağlık kontrolü endpoint'i.
    API'nin çalışır durumda olduğunu kontrol eder.
    """
    from clients.yolo_client import yolo_model_manager
    
    # Model durumlarını kontrol et
    model_status = yolo_model_manager.get_loaded_models()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=list(model_status.keys()) if model_status else [],
        models_count=len(model_status)
    )


@app.get("/api/v1/", tags=["API"])
async def api_root():
    """
    API kök endpoint - Versiyon bilgisi döndürür.
    """
    return {
        "version": "v1",
        "endpoints": {
            "inference": {
                "defect_detection": "/api/v1/detect/defects",
                "ore_classification": "/api/v1/classify/ore",
                "list_models": "/api/v1/models"
            },
            "upload": {
                "image": "/api/v1/upload/image",
                "batch": "/api/v1/upload/batch"
            }
        }
    }
