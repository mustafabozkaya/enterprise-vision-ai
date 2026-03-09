"""
YOLO Client - YOLO Model Wrapper
Ultralytics YOLO model entegrasyonu
"""

import os
import time
from typing import Dict, Optional, Any, Union
import numpy as np


class YOLOModelManager:
    """
    YOLO model yöneticisi.
    
    YOLO modellerinin yüklenmesi, önbelleğe alınması ve inference için wrapper.
    """
    
    # Varsayılan modeller
    DEFAULT_MODELS = {
        "detection": "yolo11n.pt",
        "segmentation": "yolo11n-seg.pt"
    }
    
    def __init__(self, cache_dir: str = "models"):
        """
        YOLO model manager başlatır.
        
        Args:
            cache_dir: Model önbellek dizini
        """
        self.cache_dir = cache_dir
        self._models: Dict[str, Any] = {}
        self._model_timestamps: Dict[str, float] = {}
        
        # Dizin oluştur
        os.makedirs(cache_dir, exist_ok=True)
        
        # Ultralytics'i lazy import et
        self._ultralytics = None
    
    @property
    def ultralytics(self):
        """Ultralytics modülünü lazy import eder"""
        if self._ultralytics is None:
            try:
                from ultralytics import YOLO
                self._ultralytics = YOLO
            except ImportError:
                raise ImportError(
                    "Ultralytics kütüphanesi kurulu değil. "
                    "Lütfen 'pip install ultralytics' komutuyla kurun."
                )
        return self._ultralytics
    
    async def load_model(
        self,
        model_path: str,
        device: str = "cpu",
        verbose: bool = False
    ) -> bool:
        """
        Modeli yükler.
        
        Args:
            model_path: Model dosya yolu veya model adı
            device: Cihaz ('cpu', 'cuda', 'mps')
            verbose: Detaylı çıktı
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            # Modeli oluştur
            model = self.ultralytics(model_path)
            
            # Cihazı ayarla
            model.to(device)
            
            # Önbelleğe al
            self._models[model_path] = model
            self._model_timestamps[model_path] = time.time()
            
            if verbose:
                print(f"Model yüklendi: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Modeli bellekten kaldırır.
        
        Args:
            model_name: Model adı
            
        Returns:
            bool: Başarılı ise True
        """
        if model_name in self._models:
            del self._models[model_name]
            del self._model_timestamps[model_name]
            return True
        return False
    
    def unload_all_models(self):
        """Tüm modelleri bellekten kaldırır"""
        self._models.clear()
        self._model_timestamps.clear()
    
    async def get_model(
        self,
        model_path: str,
        device: str = "cpu"
    ) -> Any:
        """
        Modeli getirir (gerekirse yükler).
        
        Args:
            model_path: Model dosya yolu veya adı
            device: Cihaz
            
        Returns:
            Any: YOLO modeli
        """
        # Önbellekte var mı kontrol et
        if model_path in self._models:
            # Zaman damgasını güncelle
            self._model_timestamps[model_path] = time.time()
            return self._models[model_path]
        
        # Modeli yükle
        await self.load_model(model_path, device)
        
        return self._models.get(model_path)
    
    def get_loaded_models(self) -> Dict[str, float]:
        """
        Yüklenmiş modelleri döndürür.
        
        Returns:
            Dict: Model adları ve yüklenme zamanları
        """
        return {
            name: self._model_timestamps.get(name, 0)
            for name in self._models.keys()
        }
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Modelin yüklü olup olmadığını kontrol eder.
        
        Args:
            model_name: Model adı
            
        Returns:
            bool: Yüklü ise True
        """
        return model_name in self._models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Model bilgilerini döndürür.
        
        Args:
            model_name: Model adı
            
        Returns:
            Optional[Dict]: Model bilgileri
        """
        if model_name not in self._models:
            return None
        
        model = self._models[model_name]
        
        return {
            "name": model_name,
            "loaded_at": self._model_timestamps.get(model_name),
            "task": getattr(model, "task", "detect"),
            "device": str(model.device)
        }


class YOLOClient:
    """
    YOLO inference client.
    
    YOLO modeli ile inference yapmak için basit wrapper.
    """
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "cpu",
        manager: Optional[YOLOModelManager] = None
    ):
        """
        YOLO client başlatır.
        
        Args:
            model_path: Model dosya yolu veya adı
            device: Cihaz
            manager: Model manager instance'ı
        """
        self.model_path = model_path
        self.device = device
        self.manager = manager or YOLOModelManager()
        self._model = None
    
    @property
    def model(self) -> Any:
        """Modeli lazy load eder"""
        if self._model is None:
            import asyncio
            self._model = asyncio.run(self.manager.get_model(self.model_path, self.device))
        return self._model
    
    def predict(
        self,
        source: Union[str, np.ndarray, Any],
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        **kwargs
    ) -> Any:
        """
        Inference yap.
        
        Args:
            source: Görüntü kaynağı
            conf: Confidence threshold
            iou: IoU threshold
            max_det: Maksimum tespit sayısı
            **kwargs: Ek parametreler
            
        Returns:
            Any: Inference sonuçları
        """
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=max_det,
            **kwargs
        )
    
    def predict_batch(
        self,
        sources: list,
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs
    ) -> Any:
        """
        Batch inference yap.
        
        Args:
            sources: Görüntü kaynakları listesi
            conf: Confidence threshold
            iou: IoU threshold
            **kwargs: Ek parametreler
            
        Returns:
            Any: Inference sonuçları
        """
        return self.model.predict(
            source=sources,
            conf=conf,
            iou=iou,
            **kwargs
        )
    
    def predict_video(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
        **kwargs
    ) -> Any:
        """
        Video inference yap.
        
        Args:
            source: Video dosya yolu veya URL
            conf: Confidence threshold
            iou: IoU threshold
            save: Sonucu kaydet
            **kwargs: Ek parametreler
            
        Returns:
            Any: Inference sonuçları
        """
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            stream=True,
            **kwargs
        )


# Singleton instance
yolo_model_manager = YOLOModelManager()
