"""
Inference Service - YOLO & HuggingFace Inference İş Mantığı
Defekt tespiti ve cevher sınıflandırma servisleri

Bu servis hem YOLO hem de HuggingFace modellerini destekler:
- YOLO: Ultralytics YOLO modelleri
- HuggingFace: SAM, DETR, YOLOS gibi modeller
"""

import base64
import io
import time
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np
from PIL import Image

from clients.yolo_client import YOLOModelManager
from clients.huggingface_client import HuggingFaceVisionClient
from api.schemas.response import DetectionResult, BoundingBox, ImageInfo, Metrics
from services.utils import (
    get_defect_colors,
    get_ore_class_colors,
    get_severity_level,
    calculate_anomaly_score,
    calculate_ore_metrics,
    calculate_metal_ratio,
)


class ModelProvider(Enum):
    """
    Desteklenen model sağlayıcıları.
    """

    YOLO = "yolo"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"  # YOLO dene, başarısız olursa HuggingFace dene


class InferenceService:
    """
    YOLO ve HuggingFace inference servisi.

    Defekt tespiti ve cevher sınıflandırma işlemlerini gerçekleştirir.
    Hem YOLO hem de HuggingFace modellerini destekler.
    """

    # Defekt sınıfları (Türkçe)
    DEFECT_CLASSES = {0: "çatlak", 1: "çizik", 2: "delik", 3: "leke", 4: "deformasyon"}

    # Cevher sınıfları (Türkçe)
    ORE_CLASSES = {0: "manyetit", 1: "krom", 2: "atık", 3: "düşük tenör"}

    def __init__(
        self,
        model_manager: YOLOModelManager,
        model_provider: ModelProvider = ModelProvider.AUTO,
        device: str = "cpu",
    ):
        """
        Inference service başlatır.

        Args:
            model_manager: YOLO model yöneticisi
            model_provider: Model sağlayıcısı (YOLO, HuggingFace veya AUTO)
            device: Cihaz ('cpu', 'cuda', 'mps')
        """
        self.model_manager = model_manager
        self.model_provider = model_provider
        self.device = device

        # HuggingFace client (lazy load)
        self._hf_client: Optional[HuggingFaceVisionClient] = None

        # Varsayılan modeller
        self.default_defect_model = "yolo11n-seg.pt"
        self.default_ore_model = "yolo11n-seg.pt"

        # HuggingFace model adları
        self.hf_defect_detection_model = "detr-resnet-50"
        self.hf_defect_segmentation_model = "sam-vit-base"
        self.hf_ore_detection_model = "detr-resnet-50"

        # Renk sabitleri
        self.defect_colors = get_defect_colors()
        self.ore_colors = get_ore_class_colors()

    @property
    def hf_client(self) -> HuggingFaceVisionClient:
        """
        HuggingFace client'ı lazy load eder.

        Returns:
            HuggingFaceVisionClient: HuggingFace vision client
        """
        if self._hf_client is None:
            self._hf_client = HuggingFaceVisionClient(
                device=self.device, cache_dir="models/huggingface"
            )
        return self._hf_client

    def _get_effective_provider(self, provider: Optional[ModelProvider]) -> ModelProvider:
        """
        Etkili model sağlayıcısını belirler.

        Args:
            provider: İstenen sağlayıcı

        Returns:
            ModelProvider: Kullanılacak sağlayıcı
        """
        if provider == ModelProvider.AUTO:
            return self.model_provider
        return provider or self.model_provider

    async def detect_defects(
        self,
        image_data: str,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        model_provider: Optional[ModelProvider] = None,
    ) -> Dict[str, Any]:
        """
        Defekt tespiti yapar.

        Args:
            image_data: Görüntü verisi (base64, URL veya dosya yolu)
            confidence_threshold: Güven eşiği
            model_name: Model adı
            iou_threshold: IoU eşiği
            max_detections: Maksimum tespit sayısı
            model_provider: Model sağlayıcısı (YOLO, HuggingFace, AUTO)

        Returns:
            Dict: Tespit sonuçları
        """
        # Görüntüyü yükle
        image = self._load_image(image_data)
        if image is None:
            raise ValueError("Görüntü yüklenemedi")

        # Etkili sağlayıcıyı belirle
        effective_provider = self._get_effective_provider(model_provider)

        # Model sağlayıcısına göre inference yap
        if effective_provider == ModelProvider.HUGGINGFACE:
            return await self._detect_defects_huggingface(
                image, confidence_threshold, model_name, max_detections
            )
        elif effective_provider == ModelProvider.YOLO:
            return await self._detect_defects_yolo(
                image,
                confidence_threshold,
                model_name or self.default_defect_model,
                iou_threshold,
                max_detections,
            )
        else:  # AUTO - önce YOLO dene, başarısız olursa HuggingFace
            try:
                return await self._detect_defects_yolo(
                    image,
                    confidence_threshold,
                    model_name or self.default_defect_model,
                    iou_threshold,
                    max_detections,
                )
            except Exception as e:
                print(f"YOLO başarısız, HuggingFace deneniyor: {e}")
                try:
                    return await self._detect_defects_huggingface(
                        image, confidence_threshold, model_name, max_detections
                    )
                except Exception as hf_error:
                    raise RuntimeError(f"Hem YOLO hem de HuggingFace başarısız: {hf_error}")

    async def _detect_defects_yolo(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        model_name: str,
        iou_threshold: float,
        max_detections: int,
    ) -> Dict[str, Any]:
        """
        YOLO ile defekt tespiti yapar.

        Args:
            image: Görüntü dizisi
            confidence_threshold: Güven eşiği
            model_name: Model adı
            iou_threshold: IoU eşiği
            max_detections: Maksimum tespit sayısı

        Returns:
            Dict: Tespit sonuçları
        """
        # Modeli yükle (veya getir)
        model = await self.model_manager.get_model(model_name)

        # Inference yap
        results = model.predict(
            image, conf=confidence_threshold, iou=iou_threshold, max_det=max_detections
        )

        # Sonuçları işle
        detections = self._process_defect_results(results, image.shape)

        # Görüntü bilgileri
        image_info = ImageInfo(
            width=image.shape[1],
            height=image.shape[0],
            channels=image.shape[2] if len(image.shape) > 2 else 1,
        )

        # Metrikleri hesapla
        metrics = self._calculate_defect_metrics(results)

        return {
            "detections": detections,
            "model_used": model_name,
            "model_provider": ModelProvider.YOLO.value,
            "image_info": image_info.model_dump(),
            "metrics": metrics.model_dump() if metrics else None,
        }

    async def _detect_defects_huggingface(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        model_name: Optional[str],
        max_detections: int,
    ) -> Dict[str, Any]:
        """
        HuggingFace (DETR + SAM) ile defekt tespiti yapar.

        Args:
            image: Görüntü dizisi
            confidence_threshold: Güven eşiği
            model_name: Model adı (kullanılmıyor, varsayılan kullanılır)
            max_detections: Maksimum tespit sayısı

        Returns:
            Dict: Tespit sonuçları
        """
        try:
            # Önce DETR ile tespit yap
            detection_model = model_name or self.hf_defect_detection_model
            vision_result = self.hf_client.detect(
                image, model_name=detection_model, confidence_threshold=confidence_threshold
            )

            # Sınıf adlarını Türkçe'ye çevir
            labels = []
            for label in vision_result.labels:
                # İngilizce'den Türkçe'ye eşleme
                label_map = {
                    "crack": "çatlak",
                    "scratch": "çizik",
                    "hole": "delik",
                    "stain": "leke",
                    "deformation": "deformasyon",
                    "defect": "defekt",
                }
                labels.append(label_map.get(label.lower(), label))

            # Sonuçları işle
            detections = []
            for i, (box, label, score) in enumerate(
                zip(vision_result.boxes, labels, vision_result.scores)
            ):
                x1, y1, x2, y2 = box

                # Severity hesapla
                severity = get_severity_level(score * 100)

                detection = DetectionResult(
                    class_name=label,
                    confidence=float(score),
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    class_id=i,
                    severity=severity,
                    metadata={"source": "huggingface", "model": detection_model},
                )

                detections.append(detection.model_dump())

            # Görüntü bilgileri
            image_info = ImageInfo(
                width=image.shape[1],
                height=image.shape[0],
                channels=image.shape[2] if len(image.shape) > 2 else 1,
            )

            # Metrikleri manuel hesapla
            metrics = Metrics(
                total_detections=len(detections),
                processing_time=0.0,
                anomaly_score=(
                    float(np.mean(vision_result.scores) * 100)
                    if len(vision_result.scores) > 0
                    else 0.0
                ),
                severity_distribution={},
            )

            return {
                "detections": detections,
                "model_used": detection_model,
                "model_provider": ModelProvider.HUGGINGFACE.value,
                "image_info": image_info.model_dump(),
                "metrics": metrics.model_dump(),
            }

        except Exception as e:
            print(f"HuggingFace defekt tespiti hatası: {e}")
            raise

    async def detect_defects_from_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        model_provider: Optional[ModelProvider] = None,
    ) -> Dict[str, Any]:
        """
        Defekt tespiti yapar (bytes olarak).

        Args:
            image_bytes: Görüntü bytes
            confidence_threshold: Güven eşiği
            model_name: Model adı
            iou_threshold: IoU eşiği
            max_detections: Maksimum tespit sayısı
            model_provider: Model sağlayıcısı

        Returns:
            Dict: Tespit sonuçları
        """
        # Bytes'tan görüntüyü yükle
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        # BGR'den RGB'ye çevir (OpenCV kullanıldığında)
        if len(image.shape) == 3 and image.shape[2] == 3:
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return await self.detect_defects(
            image_data="",  # Dummy - görüntü zaten yüklendi
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            model_provider=model_provider,
        )

    async def classify_ore(
        self,
        image_data: str,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        model_provider: Optional[ModelProvider] = None,
    ) -> Dict[str, Any]:
        """
        Cevher sınıflandırması yapar.

        Args:
            image_data: Görüntü verisi
            confidence_threshold: Güven eşiği
            model_name: Model adı
            iou_threshold: IoU eşiği
            max_detections: Maksimum tespit sayısı
            model_provider: Model sağlayıcısı

        Returns:
            Dict: Sınıflandırma sonuçları
        """
        # Görüntüyü yükle
        image = self._load_image(image_data)
        if image is None:
            raise ValueError("Görüntü yüklenemedi")

        # Etkili sağlayıcıyı belirle
        effective_provider = self._get_effective_provider(model_provider)

        # Model sağlayıcısına göre inference yap
        if effective_provider == ModelProvider.HUGGINGFACE:
            return await self._classify_ore_huggingface(
                image, confidence_threshold, model_name, max_detections
            )
        elif effective_provider == ModelProvider.YOLO:
            return await self._classify_ore_yolo(
                image,
                confidence_threshold,
                model_name or self.default_ore_model,
                iou_threshold,
                max_detections,
            )
        else:  # AUTO
            try:
                return await self._classify_ore_yolo(
                    image,
                    confidence_threshold,
                    model_name or self.default_ore_model,
                    iou_threshold,
                    max_detections,
                )
            except Exception as e:
                print(f"YOLO başarısız, HuggingFace deneniyor: {e}")
                try:
                    return await self._classify_ore_huggingface(
                        image, confidence_threshold, model_name, max_detections
                    )
                except Exception as hf_error:
                    raise RuntimeError(f"Hem YOLO hem de HuggingFace başarısız: {hf_error}")

    async def _classify_ore_yolo(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        model_name: str,
        iou_threshold: float,
        max_detections: int,
    ) -> Dict[str, Any]:
        """
        YOLO ile cevher sınıflandırması yapar.

        Args:
            image: Görüntü dizisi
            confidence_threshold: Güven eşiği
            model_name: Model adı
            iou_threshold: IoU eşiği
            max_detections: Maksimum tespit sayısı

        Returns:
            Dict: Sınıflandırma sonuçları
        """
        # Modeli yükle
        model = await self.model_manager.get_model(model_name)

        # Inference yap
        results = model.predict(
            image, conf=confidence_threshold, iou=iou_threshold, max_det=max_detections
        )

        # Sonuçları işle
        detections = self._process_ore_results(results, image.shape)

        # Görüntü bilgileri
        image_info = ImageInfo(
            width=image.shape[1],
            height=image.shape[0],
            channels=image.shape[2] if len(image.shape) > 2 else 1,
        )

        # Metrikleri hesapla
        metrics = self._calculate_ore_metrics(results)

        return {
            "detections": detections,
            "model_used": model_name,
            "model_provider": ModelProvider.YOLO.value,
            "image_info": image_info.model_dump(),
            "metrics": metrics.model_dump() if metrics else None,
        }

    async def _classify_ore_huggingface(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        model_name: Optional[str],
        max_detections: int,
    ) -> Dict[str, Any]:
        """
        HuggingFace ile cevher sınıflandırması yapar.

        Args:
            image: Görüntü dizisi
            confidence_threshold: Güven eşiği
            model_name: Model adı
            max_detections: Maksimum tespit sayısı

        Returns:
            Dict: Sınıflandırma sonuçları
        """
        try:
            detection_model = model_name or self.hf_ore_detection_model

            # DETR ile tespit yap
            vision_result = self.hf_client.detect(
                image, model_name=detection_model, confidence_threshold=confidence_threshold
            )

            # Sınıf adlarını Türkçe'ye çevir
            labels = []
            for label in vision_result.labels:
                # İngilizce'den Türkçe'ye eşleme
                label_map = {
                    "magnetite": "manyetit",
                    "chromite": "krom",
                    "waste": "atık",
                    "low grade": "düşük tenör",
                    "iron ore": "manyetit",
                    "ore": "cevher",
                }
                labels.append(label_map.get(label.lower(), label))

            # Sonuçları işle
            detections = []
            for i, (box, label, score) in enumerate(
                zip(vision_result.boxes, labels, vision_result.scores)
            ):
                x1, y1, x2, y2 = box

                detection = DetectionResult(
                    class_name=label,
                    confidence=float(score),
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    class_id=i,
                    severity=None,
                    metadata={"source": "huggingface", "model": detection_model},
                )

                detections.append(detection.model_dump())

            # Görüntü bilgileri
            image_info = ImageInfo(
                width=image.shape[1],
                height=image.shape[0],
                channels=image.shape[2] if len(image.shape) > 2 else 1,
            )

            # Sınıf dağılımını hesapla
            from collections import Counter

            class_dist = Counter(labels)

            # Metrikleri hesapla
            metrics = Metrics(
                total_detections=len(detections),
                processing_time=0.0,
                metal_ratio=float(
                    sum(1 for l in labels if l in ["manyetit", "krom"]) / max(len(labels), 1)
                ),
                class_distribution=dict(class_dist),
            )

            return {
                "detections": detections,
                "model_used": detection_model,
                "model_provider": ModelProvider.HUGGINGFACE.value,
                "image_info": image_info.model_dump(),
                "metrics": metrics.model_dump(),
            }

        except Exception as e:
            print(f"HuggingFace cevher sınıflandırması hatası: {e}")
            raise

    async def classify_ore_from_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        model_provider: Optional[ModelProvider] = None,
    ) -> Dict[str, Any]:
        """
        Cevher sınıflandırması yapar (bytes olarak).

        Args:
            image_bytes: Görüntü bytes
            confidence_threshold: Güven eşiği
            model_name: Model adı
            iou_threshold: IoU eşiği
            max_detections: Maksimum tespit sayısı
            model_provider: Model sağlayıcısı

        Returns:
            Dict: Sınıflandırma sonuçları
        """
        # Bytes'tan görüntüyü yükle
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        # BGR'den RGB'ye çevir
        if len(image.shape) == 3 and image.shape[2] == 3:
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return await self.classify_ore(
            image_data="",  # Dummy
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            model_provider=model_provider,
        )

    def _load_image(self, image_data: str) -> Optional[np.ndarray]:
        """
        Görüntü verisini yükler.

        Args:
            image_data: Görüntü verisi

        Returns:
            np.ndarray: Görüntü dizisi
        """
        try:
            # Base64 kontrolü
            if image_data.startswith("data:image"):
                # Data URL
                base64_data = image_data.split(",")[1]
                image = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image))
                return np.array(image)
            elif len(image_data) > 200 and not image_data.startswith("http"):
                # Muhtemelen base64
                image = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image))
                return np.array(image)
            elif image_data.startswith("http"):
                # URL - PIL ile aç
                image = Image.open(requests.get(image_data, stream=True).raw)
                return np.array(image)
            else:
                # Dosya yolu
                image = Image.open(image_data)
                return np.array(image)
        except Exception as e:
            print(f"Görüntü yükleme hatası: {e}")
            return None

    def _process_defect_results(self, results, image_shape: tuple) -> List[Dict]:
        """
        Defekt tespit sonuçlarını işler.

        Args:
            results: YOLO sonuçları
            image_shape: Görüntü şekli

        Returns:
            List: İşlenmiş sonuçlar
        """
        detections = []

        if results is None or len(results) == 0:
            return detections

        result = results[0]

        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)

                # Sınıf adı
                class_name = self.DEFECT_CLASSES.get(cls_id, f"defect_{cls_id}")

                # Severity hesapla
                severity = get_severity_level(conf * 100)

                detection = DetectionResult(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    class_id=cls_id,
                    severity=severity,
                    metadata={"source": "yolo"},
                )

                detections.append(detection.model_dump())

        return detections

    def _process_ore_results(self, results, image_shape: tuple) -> List[Dict]:
        """
        Cevher sınıflandırma sonuçlarını işler.

        Args:
            results: YOLO sonuçları
            image_shape: Görüntü şekli

        Returns:
            List: İşlenmiş sonuçlar
        """
        detections = []

        if results is None or len(results) == 0:
            return detections

        result = results[0]

        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)

                # Sınıf adı
                class_name = self.ORE_CLASSES.get(cls_id, f"ore_{cls_id}")

                detection = DetectionResult(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    class_id=cls_id,
                    severity=None,
                    metadata={"source": "yolo"},
                )

                detections.append(detection.model_dump())

        return detections

    def _calculate_defect_metrics(self, results) -> Optional[Metrics]:
        """
        Defekt metriklerini hesaplar.

        Args:
            results: YOLO sonuçları

        Returns:
            Metrics: Hesaplanan metrikler
        """
        if results is None or len(results) == 0:
            return None

        total_detections = 0
        severity_dist = {}

        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                total_detections += len(boxes)

                for box in boxes:
                    conf = float(box[4])
                    severity = get_severity_level(conf * 100)
                    severity_dist[severity] = severity_dist.get(severity, 0) + 1

        anomaly_score = calculate_anomaly_score(results)

        return Metrics(
            total_detections=total_detections,
            processing_time=0.0,  # İstemci tarafından hesaplanacak
            anomaly_score=anomaly_score,
            severity_distribution=severity_dist,
        )

    async def segment_with_sam(
        self, image_data: str, model_name: str = "sam-vit-base", num_points: int = 64
    ) -> Dict[str, Any]:
        """
        SAM ile segmentasyon yapar.

        Args:
            image_data: Görüntü verisi
            model_name: SAM model adı
            num_points: Kullanılacak nokta sayısı

        Returns:
            Dict: Segmentasyon sonuçları
        """
        # Görüntüyü yükle
        image = self._load_image(image_data)
        if image is None:
            raise ValueError("Görüntü yüklenemedi")

        try:
            # SAM modeli yükle
            self.hf_client.load_model("sam", model_name)

            # Maskeleri üret
            vision_result = self.hf_client.get_model("sam", model_name).generate_masks(
                image, num_points=num_points
            )

            # Sonuçları işle
            detections = []
            for i, (box, mask) in enumerate(zip(vision_result.boxes, vision_result.masks)):
                x1, y1, x2, y2 = box

                detection = DetectionResult(
                    class_name="segment",
                    confidence=1.0,  # SAM confidence yok
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    class_id=i,
                    severity=None,
                    metadata={
                        "source": "sam",
                        "model": model_name,
                        "mask_shape": mask.shape if mask is not None else None,
                    },
                )

                detections.append(detection.model_dump())

            # Görüntü bilgileri
            image_info = ImageInfo(
                width=image.shape[1],
                height=image.shape[0],
                channels=image.shape[2] if len(image.shape) > 2 else 1,
            )

            return {
                "detections": detections,
                "model_used": model_name,
                "model_provider": ModelProvider.HUGGINGFACE.value,
                "image_info": image_info.model_dump(),
                "masks": vision_result.masks.tolist() if vision_result.masks is not None else [],
            }

        except Exception as e:
            print(f"SAM segmentasyon hatası: {e}")
            raise

    def _calculate_ore_metrics(self, results) -> Optional[Metrics]:
        """
        Cevher metriklerini hesaplar.

        Args:
            results: YOLO sonuçları

        Returns:
            Metrics: Hesaplanan metrikler
        """
        if results is None or len(results) == 0:
            return None

        counts = calculate_ore_metrics(results)
        metal_ratio = calculate_metal_ratio(counts)

        return Metrics(
            total_detections=sum(counts.values()),
            processing_time=0.0,
            metal_ratio=metal_ratio,
            class_distribution=counts,
        )


# Requests modülü lazy import için
import requests
