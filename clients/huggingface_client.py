"""
HuggingFace Client - HuggingFace Entegrasyonu
HuggingFace modelleri ve Spaces entegrasyonu

Bu modül HuggingFace vision modelleri için kapsamlı destek sağlar:
- SAM (Segment Anything Model) ile instance segmentasyonu
- DETR, YOLOS gibi nesne tespit modelleri
- Maske üretimi ve işleme
"""

import os
import base64
import io
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

# Lazy imports için TypeVar
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import cv2


@dataclass
class VisionResult:
    """
    Vision model sonuçları için veri sınıfı.
    """
    boxes: np.ndarray  # [N, 4] - x1, y1, x2, y2
    labels: List[str]  # Sınıf isimleri
    scores: np.ndarray  # [N] - güven skorları
    masks: Optional[np.ndarray]  # [N, H, W] - segmentasyon maskeleri
    
    def __len__(self):
        return len(self.labels)
    
    def to_dict(self) -> Dict:
        """Sonuçları dict olarak döndürür."""
        return {
            "boxes": self.boxes.tolist() if self.boxes is not None else [],
            "labels": self.labels,
            "scores": self.scores.tolist() if self.scores is not None else [],
            "masks": self.masks.tolist() if self.masks is not None else []
        }


class SAMModel:
    """
    Segment Anything Model (SAM) wrapper.
    
    Facebook'un SAM modeli ile instance segmentasyonu sağlar.
    """
    
    # Desteklenen SAM modelleri
    SAM_MODELS = {
        "sam-vit-base": "facebook/sam-vit-base",
        "sam-vit-large": "facebook/sam-vit-large",
        "sam-vit-huge": "facebook/sam-vit-huge",
        "sam2-base": "facebook/sam2-base",
        "sam2-large": "facebook/sam2-large"
    }
    
    def __init__(
        self,
        model_id: str = "sam-vit-base",
        device: str = "cpu",
        cache_dir: str = "models/huggingface"
    ):
        """
        SAM model başlatır.
        
        Args:
            model_id: Model ID veya model adı
            device: Cihaz ('cpu', 'cuda', 'mps')
            cache_dir: Önbellek dizini
        """
        self.model_id = self.SAM_MODELS.get(model_id, model_id)
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _lazy_imports(self):
        """Gerekli modülleri lazy olarak import eder."""
        try:
            import torch
            from transformers import SamProcessor, SamModel
            return torch, SamProcessor, SamModel
        except ImportError as e:
            raise ImportError(
                "transformers ve torch kütüphaneleri kurulu değil. "
                "Lütfen 'pip install torch transformers' komutuyla kurun."
            ) from e
    
    def load(self, force: bool = False) -> bool:
        """
        Modeli yükler.
        
        Args:
            force: Yeniden yükleme zorla
            
        Returns:
            bool: Başarılı ise True
        """
        if self._model is not None and not force:
            return True
            
        try:
            torch, SamProcessor, SamModel = self._lazy_imports()
            
            # Processor'ı yükle
            self._processor = SamProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            # Modeli yükle
            self._model = SamModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            # Cihazı ayarla
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            elif self.device == "mps":
                try:
                    self._model = self._model.to("mps")
                except:
                    self._model = self._model.to("cpu")
            else:
                self._model = self._model.to("cpu")
            
            self._model.eval()
            return True
            
        except Exception as e:
            print(f"SAM model yükleme hatası: {e}")
            return False
    
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        multimask_output: bool = False
    ) -> VisionResult:
        """
        Segmentasyon yapar.
        
        Args:
            image: Görüntü
            points: Noktalar [N, 2] (x, y)
            labels: Nokta etiketleri [N] (1: foreground, 0: background)
            multimask_output: Çoklu maske çıktısı
            
        Returns:
            VisionResult: Segmentasyon sonuçları
        """
        if self._model is None:
            self.load()
        
        import torch
        
        # Görüntüyü işle
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Otomatik nokta üretimi (kılavuz noktalar)
        if points is None:
            # Görüntüyü ızgara olarak böl ve her hücre için bir nokta al
            w, h = image.size
            grid_size = 8
            points = []
            labels = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x = int((i + 0.5) * w / grid_size)
                    y = int((j + 0.5) * h / grid_size)
                    points.append([x, y])
                    labels.append(1)  # Hepsi foreground
            
            points = np.array(points)
            labels = np.array(labels)
        
        # Girdileri hazırla
        input_points = torch.tensor(points).unsqueeze(0)
        input_labels = torch.tensor(labels).unsqueeze(0)
        
        inputs = self._processor(
            image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        )
        
        # Cihaza taşı
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Maskeleri işle
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )
        
        # En iyi maskeyi seç
        import torch.nn.functional as F
        
        mask = masks[0][0]  # İlk batch
        
        if multimask_output and len(mask) > 1:
            # IoU'ya göre sırala
            scores = []
            for m in mask:
                scores.append(m.sum().item())
            idx = np.argmax(scores)
            mask = mask[idx:idx+1]
        else:
            mask = mask[0:1]  # İlk maske
        
        # Maskeyi numpy array'e çevir
        mask_np = mask.squeeze().cpu().numpy()
        
        # Bounding box hesapla
        if mask_np.ndim == 2:
            mask_np = mask_np[np.newaxis, ...]
        
        boxes = []
        valid_masks = []
        
        for m in mask_np:
            if m.sum() > 0:
                # Maskeden bounding box hesapla
                rows = np.any(m, axis=1)
                cols = np.any(m, axis=0)
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                boxes.append([x1, y1, x2, y2])
                valid_masks.append(m)
        
        if len(boxes) == 0:
            return VisionResult(
                boxes=np.array([]).reshape(0, 4),
                labels=[],
                scores=np.array([]),
                masks=None
            )
        
        return VisionResult(
            boxes=np.array(boxes),
            labels=["segment"] * len(boxes),
            scores=np.ones(len(boxes)),
            masks=np.array(valid_masks) if valid_masks else None
        )
    
    def generate_masks(
        self,
        image: Union[str, Image.Image, np.ndarray],
        num_points: int = 64
    ) -> VisionResult:
        """
        Görüntüden otomatik maskeler üretir.
        
        Args:
            image: Görüntü
            num_points: Üretilecek nokta sayısı
            
        Returns:
            VisionResult: Maske sonuçları
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        w, h = image.size
        
        # Izgara noktaları oluştur
        grid = int(np.sqrt(num_points))
        points = []
        labels = []
        
        for i in range(grid):
            for j in range(grid):
                x = int((i + 0.5) * w / grid)
                y = int((j + 0.5) * h / grid)
                points.append([x, y])
                labels.append(1)
        
        return self.predict(
            image,
            np.array(points),
            np.array(labels),
            multimask_output=False
        )
    
    def __call__(self, image, **kwargs):
        """Kısayol metodu."""
        return self.predict(image, **kwargs)


class DetectionModel:
    """
    HuggingFace nesne tespit modeli wrapper.
    
    DETR, YOLOS gibi modelleri destekler.
    """
    
    # Desteklenen modeller
    DETECTION_MODELS = {
        "detr-resnet-50": "facebook/detr-resnet-50",
        "detr-resnet-101": "facebook/detr-resnet-101",
        "detr-large": "facebook/detr-large",
        "yolos-small": "hustvl/yolos-small",
        "yolos-tiny": "hustvl/yolos-tiny",
        "yolos-base": "hustvl/yolos-base",
    }
    
    def __init__(
        self,
        model_id: str = "detr-resnet-50",
        device: str = "cpu",
        cache_dir: str = "models/huggingface"
    ):
        """
        Detection model başlatır.
        
        Args:
            model_id: Model ID veya model adı
            device: Cihaz
            cache_dir: Önbellek dizini
        """
        self.model_id = self.DETECTION_MODELS.get(model_id, model_id)
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _lazy_imports(self):
        """Gerekli modülleri lazy olarak import eder."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            return torch, AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as e:
            raise ImportError(
                "transformers ve torch kütüphaneleri kurulu değil. "
                "Lütfen 'pip install torch transformers' komutuyla kurun."
            ) from e
    
    def load(self, force: bool = False) -> bool:
        """
        Modeli yükler.
        
        Args:
            force: Yeniden yükleme zorla
            
        Returns:
            bool: Başarılı ise True
        """
        if self._model is not None and not force:
            return True
            
        try:
            torch, AutoImageProcessor, AutoModelForObjectDetection = self._lazy_imports()
            
            # Processor'ı yükle
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            # Modeli yükle
            self._model = AutoModelForObjectDetection.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            # Cihazı ayarla
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            else:
                self._model = self._model.to("cpu")
            
            self._model.eval()
            return True
            
        except Exception as e:
            print(f"Detection model yükleme hatası: {e}")
            return False
    
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ) -> VisionResult:
        """
        Nesne tespiti yapar.
        
        Args:
            image: Görüntü
            confidence_threshold: Güven eşiği
            nms_threshold: NMS eşiği
            
        Returns:
            VisionResult: Tespit sonuçları
        """
        if self._model is None:
            self.load()
        
        import torch
        
        # Görüntüyü işle
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Girdiyi hazırla
        inputs = self._processor(images=image, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Sonuçları işle
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]
        
        # NMS uygula
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        # Etiketleri model config'den al
        label_names = self._model.config.id2label if hasattr(self._model.config, "id2label") else {}
        
        # Filtrele
        keep = scores >= confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        label_names_list = [label_names.get(l, f"class_{l}") for l in labels]
        
        return VisionResult(
            boxes=boxes,
            labels=label_names_list,
            scores=scores,
            masks=None  # Detection modelleri mask üretmez
        )
    
    def __call__(self, image, **kwargs):
        """Kısayol metodu."""
        return self.predict(image, **kwargs)


class HuggingFaceVisionClient:
    """
    HuggingFace Vision modelleri için kapsamlı client.
    
    SAM, DETR, YOLOS gibi modelleri destekler ve
    instance segmentasyonu, nesne tespiti sağlar.
    """
    
    # Model türleri
    MODEL_TYPES = {
        "sam": {
            "default": "sam-vit-base",
            "class": SAMModel
        },
        "detection": {
            "default": "detr-resnet-50",
            "class": DetectionModel
        }
    }
    
    def __init__(
        self,
        device: str = "cpu",
        cache_dir: str = "models/huggingface"
    ):
        """
        HuggingFace vision client başlatır.
        
        Args:
            device: Cihaz ('cpu', 'cuda', 'mps')
            cache_dir: Önbellek dizini
        """
        self.device = device
        self.cache_dir = cache_dir
        self._models: Dict[str, Any] = {}
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_model(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """
        Modeli yükler.
        
        Args:
            model_type: Model türü ('sam', 'detection')
            model_name: Model adı
            force: Yeniden yükleme zorla
            
        Returns:
            bool: Başarılı ise True
        """
        model_config = self.MODEL_TYPES.get(model_type)
        if model_config is None:
            print(f"Bilinmeyen model türü: {model_type}")
            return False
        
        model_name = model_name or model_config["default"]
        model_key = f"{model_type}_{model_name}"
        
        # Önbellekte var mı kontrol et
        if model_key in self._models and not force:
            return True
        
        # Modeli oluştur ve yükle
        model_class = model_config["class"]
        model = model_class(
            model_id=model_name,
            device=self.device,
            cache_dir=self.cache_dir
        )
        
        success = model.load(force=force)
        if success:
            self._models[model_key] = model
        
        return success
    
    def get_model(self, model_type: str, model_name: Optional[str] = None) -> Any:
        """
        Yüklenmiş modeli getirir.
        
        Args:
            model_type: Model türü
            model_name: Model adı
            
        Returns:
            Any: Model
        """
        model_config = self.MODEL_TYPES.get(model_type)
        if model_config is None:
            return None
        
        model_name = model_name or model_config["default"]
        model_key = f"{model_type}_{model_name}"
        
        if model_key not in self._models:
            self.load_model(model_type, model_name)
        
        return self._models.get(model_key)
    
    def segment(
        self,
        image: Union[str, Image.Image, np.ndarray],
        model_name: str = "sam-vit-base",
        confidence_threshold: float = 0.5
    ) -> VisionResult:
        """
        Instance segmentasyon yapar.
        
        Args:
            image: Görüntü
            model_name: Model adı
            confidence_threshold: Güven eşiği
            
        Returns:
            VisionResult: Segmentasyon sonuçları
        """
        model = self.get_model("sam", model_name)
        if model is None:
            raise RuntimeError("SAM modeli yüklenemedi")
        
        return model.predict(image)
    
    def detect(
        self,
        image: Union[str, Image.Image, np.ndarray],
        model_name: str = "detr-resnet-50",
        confidence_threshold: float = 0.5
    ) -> VisionResult:
        """
        Nesne tespiti yapar.
        
        Args:
            image: Görüntü
            model_name: Model adı
            confidence_threshold: Güven eşiği
            
        Returns:
            VisionResult: Tespit sonuçları
        """
        model = self.get_model("detection", model_name)
        if model is None:
            raise RuntimeError("Detection modeli yüklenemedi")
        
        return model.predict(image, confidence_threshold=confidence_threshold)
    
    def segment_and_detect(
        self,
        image: Union[str, Image.Image, np.ndarray],
        detection_model: str = "detr-resnet-50",
        segmentation_model: str = "sam-vit-base",
        confidence_threshold: float = 0.5
    ) -> VisionResult:
        """
        Önce tespit, sonra segmentasyon yapar.
        
        Args:
            image: Görüntü
            detection_model: Tespit modeli
            segmentation_model: Segmentasyon modeli
            confidence_threshold: Güven eşiği
            
        Returns:
            VisionResult: Birleşik sonuçlar
        """
        # Önce nesne tespiti yap
        detections = self.detect(
            image,
            model_name=detection_model,
            confidence_threshold=confidence_threshold
        )
        
        # Her tespit için segmentasyon maskeleri üret
        if detections.masks is not None and len(detections.masks) > 0:
            return detections
        
        # SAM ile maskeleri yeniden üret
        sam_model = self.get_model("sam", segmentation_model)
        if sam_model is None:
            return detections
        
        # Her bounding box için maske üret
        masks = []
        for box in detections.boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Kılavuz noktası olarak merkez noktasını kullan
            sam_result = sam_model.predict(
                image,
                points=np.array([[center_x, center_y]]),
                labels=np.array([1])
            )
            
            if sam_result.masks is not None and len(sam_result.masks) > 0:
                masks.append(sam_result.masks[0])
            else:
                # Bounding box'tan basit maske oluştur
                if isinstance(image, Image.Image):
                    w, h = image.size
                else:
                    h, w = image.shape[:2]
                
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
                masks.append(mask)
        
        if masks:
            detections.masks = np.array(masks)
        
        return detections
    
    def unload_all(self):
        """Tüm modelleri bellekten kaldırır."""
        self._models.clear()
    
    def get_loaded_models(self) -> List[str]:
        """
        Yüklenmiş modelleri döndürür.
        
        Returns:
            List[str]: Model listesi
        """
        return list(self._models.keys())


# Önceki sınıflar korunuyor...


class HuggingFaceClient:
    """
    HuggingFace client.
    
    HuggingFace modelleri ile entegrasyon sağlar.
    """
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        cache_dir: str = "models/huggingface"
    ):
        """
        HuggingFace client başlatır.
        
        Args:
            api_token: HuggingFace API token
            cache_dir: Önbellek dizini
        """
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.cache_dir = cache_dir
        self._client = None
        
        # Dizin oluştur
        os.makedirs(cache_dir, exist_ok=True)
    
    @property
    def client(self):
        """HuggingFace Hub client'ı lazy load eder"""
        if self._client is None:
            try:
                from huggingface_hub import HfApi
                self._client = HfApi(token=self.api_token)
            except ImportError:
                raise ImportError(
                    "huggingface_hub kütüphanesi kurulu değil. "
                    "Lütfen 'pip install huggingface_hub' komutuyla kurun."
                )
        return self._client
    
    def list_models(
        self,
        filter: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        HuggingFace modellerini listeler.
        
        Args:
            filter: Filtre
            search: Arama terimi
            limit: Sonuç sayısı
            
        Returns:
            List[Dict]: Model bilgileri
        """
        try:
            from huggingface_hub import list_models
            
            models = list_models(
                filter=filter,
                search=search,
                limit=limit,
                token=self.api_token
            )
            
            return [
                {
                    "id": model.id,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags
                }
                for model in models
            ]
        except Exception as e:
            print(f"Model listeleme hatası: {e}")
            return []
    
    def download_model(
        self,
        model_id: str,
        revision: Optional[str] = None
    ) -> str:
        """
        Modeli indirir.
        
        Args:
            model_id: Model ID (örn: "facebook/detr-resnet-50")
            revision: Model revizyonu
            
        Returns:
            str: İndirilen model yolu
        """
        try:
            from huggingface_hub import snapshot_download
            
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                token=self.api_token
            )
            
            return model_path
            
        except Exception as e:
            print(f"Model indirme hatası: {e}")
            raise
    
    def upload_model(
        self,
        model_path: str,
        repo_id: str,
        commit_message: str = "Upload model"
    ) -> str:
        """
        Modeli yükler.
        
        Args:
            model_path: Model dosya yolu
            repo_id: Repo ID
            commit_message: Commit mesajı
            
        Returns:
            str: Repo URL
        """
        try:
            self.client.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message
            )
            
            return f"https://huggingface.co/{repo_id}"
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            raise


class HuggingFaceInferenceClient:
    """
    HuggingFace Inference API client.
    
    HuggingFace Inference API ile inference yapmak için wrapper.
    """
    
    # Desteklenen görevler
    SUPPORTED_TASKS = [
        "image-classification",
        "object-detection",
        "image-segmentation",
        "zero-shot-image-classification"
    ]
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        model_id: str = "facebook/detr-resnet-50"
    ):
        """
        Inference client başlatır.
        
        Args:
            api_token: HuggingFace API token
            model_id: Model ID
        """
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.model_id = model_id
        self._client = None
    
    @property
    def client(self):
        """Inference client'ı lazy load eder"""
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    model=self.model_id,
                    token=self.api_token
                )
            except ImportError:
                raise ImportError(
                    "huggingface_hub kütüphanesi kurulu değil. "
                    "Lütfen 'pip install huggingface_hub' komutuyla kurun."
                )
        return self._client
    
    def classify_image(
        self,
        image: Union[str, Image.Image, bytes],
        candidate_labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Görüntü sınıflandırması yapar.
        
        Args:
            image: Görüntü (dosya yolu, PIL Image veya bytes)
            candidate_labels: Aday etiketler
            
        Returns:
            List[Dict]: Sınıflandırma sonuçları
        """
        try:
            # PIL Image'ı bytes'a çevir
            if isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image = buffer.getvalue()
            
            result = self.client.zero_shot_image_classification(
                image=image,
                candidate_labels=candidate_labels
            )
            
            return [
                {
                    "label": item.label,
                    "score": item.score
                }
                for item in result
            ]
            
        except Exception as e:
            print(f"Sınıflandırma hatası: {e}")
            return []
    
    def detect_objects(
        self,
        image: Union[str, Image.Image, bytes]
    ) -> List[Dict[str, Any]]:
        """
        Nesne tespiti yapar.
        
        Args:
            image: Görüntü
            
        Returns:
            List[Dict]: Tespit sonuçları
        """
        try:
            # PIL Image'ı bytes'a çevir
            if isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image = buffer.getvalue()
            
            result = self.client.object_detection(image)
            
            return [
                {
                    "label": item.label,
                    "score": item.score,
                    "box": {
                        "xmin": item.box.xmin,
                        "ymin": item.box.ymin,
                        "xmax": item.box.xmax,
                        "ymax": item.box.ymax
                    }
                }
                for item in result
            ]
            
        except Exception as e:
            print(f"Nesne tespiti hatası: {e}")
            return []
    
    def segment_image(
        self,
        image: Union[str, Image.Image, bytes]
    ) -> List[Dict[str, Any]]:
        """
        Görüntü segmentasyonu yapar.
        
        Args:
            image: Görüntü
            
        Returns:
            List[Dict]: Segmentasyon sonuçları
        """
        try:
            # PIL Image'ı bytes'a çevir
            if isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image = buffer.getvalue()
            
            # Not: Semantik segmentasyon her modelde desteklenmeyebilir
            result = self.client.image_segmentation(image)
            
            return [
                {
                    "label": item.label,
                    "score": item.score,
                    "mask": item.mask  # Base64 encoded
                }
                for item in result
            ]
            
        except Exception as e:
            print(f"Segmentasyon hatası: {e}")
            return []


class HuggingFaceSpacesClient:
    """
    HuggingFace Spaces client.
    
    HuggingFace Spaces'de çalışan uygulamalarla entegrasyon sağlar.
    """
    
    def __init__(
        self,
        space_id: str,
        api_token: Optional[str] = None
    ):
        """
        Spaces client başlatır.
        
        Args:
            space_id: Space ID (örn: "username/space-name")
            api_token: HuggingFace API token
        """
        self.space_id = space_id
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.base_url = f"https://huggingface.co/spaces/{space_id}"
    
    def predict(
        self,
        data: Dict[str, Any],
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Space'e istek gönderir.
        
        Args:
            data: Gönderilecek veri
            timeout: Zaman aşımı süresi
            
        Returns:
            Dict: Yanıt
        """
        try:
            import requests
            
            # Space'in API endpoint'ini kullan
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=data,
                timeout=timeout,
                headers={"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Space isteği hatası: {e}")
            return {"error": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """
        Space bilgilerini getirir.
        
        Returns:
            Dict: Space bilgileri
        """
        try:
            import requests
            
            response = requests.get(
                f"https://huggingface.co/api/spaces/{self.space_id}",
                headers={"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Space bilgisi hatası: {e}")
            return {"error": str(e)}


# Singleton instance
huggingface_client = HuggingFaceClient()
