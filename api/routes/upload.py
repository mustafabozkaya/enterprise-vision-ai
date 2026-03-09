"""
API Routes - Upload Endpoints
Image Upload & Batch Processing API Endpoints
"""

import os
import time
import uuid
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
import aiofiles

from api.schemas.response import (
    UploadResponse,
    BatchUploadResponse,
    UploadedFile,
    ErrorResponse
)


router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Yükleme dizini yapılandırması
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Maksimum dosya boyutu (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# İzin verilen dosya tipleri
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"]


def validate_file(file: UploadFile) -> bool:
    """
    Dosya tipini ve boyutunu doğrular.
    
    Args:
        file: Yüklenecek dosya
        
    Returns:
        bool: Geçerli ise True
        
    Raises:
        HTTPException: Geçersiz dosya durumunda
    """
    # Dosya adını kontrol et
    filename = file.filename.lower()
    file_ext = Path(filename).suffix
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Desteklenmeyen dosya tipi: {file_ext}. İzin verilen: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # MIME tipini kontrol et
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Desteklenmeyen MIME tipi: {file.content_type}"
        )
    
    return True


async def save_uploaded_file(file: UploadFile, subdir: str = "images") -> str:
    """
    Yüklenen dosyayı kaydeder.
    
    Args:
        file: Yüklenecek dosya
        subdir: Alt dizin adı
        
    Returns:
        str: Kaydedilen dosyanın yolu
    """
    # Benzersiz dosya adı oluştur
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Dizin yapısını oluştur
    save_dir = UPLOAD_DIR / subdir
    save_dir.mkdir(exist_ok=True, parents=True)
    
    file_path = save_dir / unique_filename
    
    # Dosyayı kaydet
    async with aiofiles.open(file_path, 'wb') as f:
        contents = await file.read()
        
        # Boyut kontrolü
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Dosya boyutu çok büyük. Maksimum: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        await f.write(contents)
    
    return str(file_path)


@router.post(
    "/upload/image",
    response_model=UploadResponse,
    summary="Upload Single Image",
    description="Upload a single image for processing"
)
@limiter.limit("30/minute")
async def upload_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to upload"),
    subdir: Optional[str] = File("images", description="Subdirectory to save to")
):
    """
    Tek görüntü yükle.
    
    İşlenmek üzere tek bir görüntü dosyası yükler.
    
    - **file**: Yüklenecek görüntü dosyası (JPEG, PNG, WebP, BMP)
    - **subdir**: Kaydedilecek alt dizin
    """
    try:
        start_time = time.time()
        
        # Dosyayı doğrula
        validate_file(file)
        
        # Dosyayı kaydet
        file_path = await save_uploaded_file(file, subdir)
        
        # Orijinal dosya adını al
        original_filename = file.filename
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            success=True,
            file=UploadedFile(
                filename=original_filename,
                stored_path=file_path,
                size=os.path.getsize(file_path),
                content_type=file.content_type
            ),
            message="Görüntü başarıyla yüklendi.",
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dosya yüklenirken hata oluştu: {str(e)}"
        )


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    summary="Upload Batch of Images",
    description="Upload multiple images for batch processing"
)
@limiter.limit("10/minute")
async def upload_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="List of image files to upload"),
    subdir: Optional[str] = File("batch", description="Subdirectory to save to")
):
    """
    Toplu görüntü yükle.
    
    Batch processing için birden fazla görüntü dosyası yükler.
    
    - **files**: Yüklenecek görüntü dosyaları listesi (max 20)
    - **subdir**: Kaydedilecek alt dizin
    """
    try:
        start_time = time.time()
        
        # Maksimum dosya sayısı kontrolü
        if len(files) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maksimum 20 dosya yükleyebilirsiniz."
            )
        
        uploaded_files = []
        errors = []
        
        # Her dosyayı işle
        for file in files:
            try:
                # Dosyayı doğrula
                validate_file(file)
                
                # Dosyayı kaydet
                file_path = await save_uploaded_file(file, subdir)
                
                uploaded_files.append(
                    UploadedFile(
                        filename=file.filename,
                        stored_path=file_path,
                        size=os.path.getsize(file_path),
                        content_type=file.content_type
                    )
                )
                
            except HTTPException as e:
                errors.append({
                    "filename": file.filename,
                    "error": e.detail
                })
            except Exception as e:
                errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        # Sonuçları döndür
        success_count = len(uploaded_files)
        error_count = len(errors)
        
        return BatchUploadResponse(
            success=True,
            files=uploaded_files,
            total_count=len(files),
            success_count=success_count,
            error_count=error_count,
            errors=errors if errors else None,
            message=f"{success_count} dosya başarıyla yüklendi." + 
                   (f" {error_count} dosya hatalı." if error_count else ""),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Toplu yükleme sırasında hata oluştu: {str(e)}"
        )


@router.get(
    "/upload/files",
    summary="List Uploaded Files",
    description="List all uploaded files"
)
@limiter.limit("30/minute")
async def list_uploaded_files(
    request: Request,
    subdir: Optional[str] = None
):
    """
    Yüklenen dosyaları listele.
    
    Belirtilen alt dizindeki tüm dosyaları listeler.
    """
    try:
        if subdir:
            upload_path = UPLOAD_DIR / subdir
        else:
            upload_path = UPLOAD_DIR
        
        if not upload_path.exists():
            return {"files": [], "count": 0}
        
        files = []
        for file_path in upload_path.rglob("*"):
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        
        return {
            "files": files,
            "count": len(files),
            "directory": str(upload_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dosya listelenirken hata oluştu: {str(e)}"
        )


@router.delete(
    "/upload/files",
    summary="Delete Uploaded File",
    description="Delete a specific uploaded file"
)
@limiter.limit("30/minute")
async def delete_uploaded_file(
    request: Request,
    file_path: str
):
    """
    Yüklenen dosyayı sil.
    
    Belirtilen dosyayı sunucudan siler.
    """
    try:
        # Güvenlik kontrolü - sadece uploads dizini içindeki dosyaları sil
        full_path = Path(file_path)
        
        if not str(full_path).startswith(str(UPLOAD_DIR.resolve())):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz dosya yolu."
            )
        
        if not full_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dosya bulunamadı."
            )
        
        # Dosyayı sil
        full_path.unlink()
        
        return {
            "success": True,
            "message": f"{full_path.name} dosyası silindi."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dosya silinirken hata oluştu: {str(e)}"
        )
