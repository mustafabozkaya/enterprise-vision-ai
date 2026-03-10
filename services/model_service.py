"""
Model Service - Model Yönetimi
YOLO modellerinin yüklenmesi, kaldırılması ve listelenmesi
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.schemas.response import ModelInfo


class ModelService:
    """
    Model yönetim servisi.

    YOLO modellerinin yaşam döngüsünü yönetir.
    """

    # Mevcut modeller
    AVAILABLE_MODELS = {
        "yolo11n.pt": {
            "type": "detection",
            "path": "models/yolo11n.pt",
            "task_type": "detection",
            "classes": ["defect", "ore"],
            "size_mb": 6.2,
        },
        "yolo11n-seg.pt": {
            "type": "segmentation",
            "path": "models/yolo11n-seg.pt",
            "task_type": "segmentation",
            "classes": [
                "çatlak",
                "çizik",
                "delik",
                "leke",
                "deformasyon",
                "manyetit",
                "krom",
                "atık",
                "düşük tenör",
            ],
            "size_mb": 7.8,
        },
        "yolo11s.pt": {
            "type": "detection",
            "path": "models/yolo11s.pt",
            "task_type": "detection",
            "classes": ["defect", "ore"],
            "size_mb": 21.4,
        },
        "yolo11s-seg.pt": {
            "type": "segmentation",
            "path": "models/yolo11s-seg.pt",
            "task_type": "segmentation",
            "classes": [
                "çatlak",
                "çizik",
                "delik",
                "leke",
                "deformasyon",
                "manyetit",
                "krom",
                "atık",
                "düşük tenör",
            ],
            "size_mb": 26.9,
        },
        "yolo11m.pt": {
            "type": "detection",
            "path": "models/yolo11m.pt",
            "task_type": "detection",
            "classes": ["defect", "ore"],
            "size_mb": 51.4,
        },
    }

    # Defekt model sınıfları (Türkçe)
    DEFECT_CLASSES = [
        "çatlak",  # Crack
        "çizik",  # Scratch
        "delik",  # Hole
        "leke",  # Stain
        "deformasyon",  # Deformation
    ]

    # Cevher model sınıfları (Türkçe)
    ORE_CLASSES = [
        "manyetit",  # Magnetite
        "krom",  # Chromite
        "atık",  # Waste
        "düşük tenör",  # Low grade
    ]

    def __init__(self, model_manager):
        """
        Model service başlatır.

        Args:
            model_manager: YOLO model yöneticisi
        """
        self.model_manager = model_manager

    async def list_models(
        self, include_loaded_only: bool = False, model_type: Optional[str] = None
    ) -> List[ModelInfo]:
        """
        Mevcut modelleri listeler.

        Args:
            include_loaded_only: Sadece yüklenmiş modelleri dahil et
            model_type: Model türü filtresi

        Returns:
            List[ModelInfo]: Model bilgileri listesi
        """
        models = []
        loaded_models = self.model_manager.get_loaded_models()

        for name, config in self.AVAILABLE_MODELS.items():
            # Filtreleme
            if model_type and config.get("type") != model_type:
                continue

            # Yüklü durumu kontrol et
            is_loaded = name in loaded_models

            if include_loaded_only and not is_loaded:
                continue

            # Sınıfları belirle
            if "seg" in name:
                classes = self.DEFECT_CLASSES + self.ORE_CLASSES
            else:
                classes = config.get("classes", [])

            # Model boyutunu kontrol et
            model_path = config.get("path", f"models/{name}")
            size_mb = config.get("size_mb")

            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)

            model_info = ModelInfo(
                name=name,
                type=config.get("type", "detection"),
                path=model_path,
                loaded=is_loaded,
                size_mb=size_mb,
                task_type=config.get("task_type", "detection"),
                classes=classes,
                metadata={
                    "description": f"YOLO11 {'segmentation' if 'seg' in name else 'detection'} model",
                    "framework": "ultralytics",
                },
            )

            models.append(model_info)

        return models

    async def load_model(self, model_name: str) -> bool:
        """
        Modeli yükler.

        Args:
            model_name: Model adı

        Returns:
            bool: Başarılı ise True
        """
        # Model mevcut mu kontrol et
        if model_name not in self.AVAILABLE_MODELS:
            # Ultralytics'ten indirmeye çalış
            try:
                return await self.model_manager.load_model(model_name)
            except Exception as e:
                print(f"Model indirme hatası: {e}")
                return False

        model_path = self.AVAILABLE_MODELS[model_name].get("path")

        return await self.model_manager.load_model(model_path)

    async def unload_model(self, model_name: str) -> bool:
        """
        Modeli bellekten kaldırır.

        Args:
            model_name: Model adı

        Returns:
            bool: Başarılı ise True
        """
        return await self.model_manager.unload_model(model_name)

    def get_loaded_models(self) -> Dict[str, Any]:
        """
        Yüklenmiş modelleri döndürür.

        Returns:
            Dict: Yüklenmiş modeller
        """
        return self.model_manager.get_loaded_models()

    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Model bilgilerini döndürür.

        Args:
            model_name: Model adı

        Returns:
            Optional[ModelInfo]: Model bilgileri
        """
        if model_name not in self.AVAILABLE_MODELS:
            return None

        config = self.AVAILABLE_MODELS[model_name]
        loaded_models = self.model_manager.get_loaded_models()

        return ModelInfo(
            name=model_name,
            type=config.get("type", "detection"),
            path=config.get("path"),
            loaded=model_name in loaded_models,
            size_mb=config.get("size_mb"),
            task_type=config.get("task_type", "detection"),
            classes=config.get("classes", []),
        )

    def validate_model_name(self, model_name: str) -> bool:
        """
        Model adının geçerli olup olmadığını kontrol eder.

        Args:
            model_name: Model adı

        Returns:
            bool: Geçerli ise True
        """
        return model_name in self.AVAILABLE_MODELS or model_name.endswith(".pt")
