"""
Inference Service - YOLO inference wrapper for FastAPI endpoints.
"""

import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from enterprise_vision_ai.clients.yolo_client import YOLOModelManager

DEFAULT_MODEL = "yolo11n-seg.pt"


class InferenceService:
    """Wraps YOLO inference for FastAPI endpoints."""

    def __init__(self, manager: YOLOModelManager):
        self.manager = manager

    async def _bytes_to_array(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)

    async def _b64_to_array(self, image_data: str) -> np.ndarray:
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        return await self._bytes_to_array(base64.b64decode(image_data))

    def _parse_results(self, results: Any, model_name: str) -> Dict:
        detections: List[Dict] = []
        if results:
            for r in results:
                if hasattr(r, "boxes") and r.boxes is not None:
                    boxes = r.boxes.data.cpu().numpy()
                    names = getattr(r, "names", {})
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls_id = (
                            float(box[0]), float(box[1]),
                            float(box[2]), float(box[3]),
                            float(box[4]), int(box[5]),
                        )
                        detections.append({
                            "class_name": names.get(cls_id, str(cls_id)),
                            "class_id": cls_id,
                            "confidence": round(conf, 4),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        })
        return {
            "detections": detections,
            "model_used": model_name,
            "image_info": {"detection_count": len(detections)},
        }

    async def detect_defects(
        self,
        image_data: str,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
    ) -> Dict:
        model_path = model_name or DEFAULT_MODEL
        image = await self._b64_to_array(image_data)
        model = await self.manager.get_model(model_path)
        results = model.predict(source=image, conf=confidence_threshold, verbose=False)
        return self._parse_results(results, model_path)

    async def detect_defects_from_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
    ) -> Dict:
        model_path = model_name or DEFAULT_MODEL
        image = await self._bytes_to_array(image_bytes)
        model = await self.manager.get_model(model_path)
        results = model.predict(source=image, conf=confidence_threshold, verbose=False)
        return self._parse_results(results, model_path)

    async def classify_ore(
        self,
        image_data: str,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
    ) -> Dict:
        return await self.detect_defects(image_data, confidence_threshold, model_name)

    async def classify_ore_from_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
    ) -> Dict:
        return await self.detect_defects_from_bytes(image_bytes, confidence_threshold, model_name)
