"""
Model Service - Model management wrapper for FastAPI endpoints.
"""

from typing import Dict, List

from enterprise_vision_ai.clients.yolo_client import YOLOModelManager

AVAILABLE_MODELS = [
    {"name": "yolo11n-seg.pt", "version": "11n", "type": "segmentation"},
    {"name": "yolo11s-seg.pt", "version": "11s", "type": "segmentation"},
    {"name": "yolo11n.pt", "version": "11n", "type": "detection"},
]


class ModelService:
    """Wraps YOLOModelManager for FastAPI model management endpoints."""

    def __init__(self, manager: YOLOModelManager):
        self.manager = manager

    def get_loaded_models(self) -> Dict:
        return self.manager.get_loaded_models()

    async def list_models(self) -> List[Dict]:
        loaded = self.manager.get_loaded_models()
        return [
            {
                "name": m["name"],
                "type": m["type"],
                "task_type": m["type"],
                "loaded": m["name"] in loaded,
                "path": m["name"],
                "metadata": {"version": m["version"]},
            }
            for m in AVAILABLE_MODELS
        ]

    async def load_model(self, model_name: str) -> bool:
        return await self.manager.load_model(model_name)

    async def unload_model(self, model_name: str) -> bool:
        return await self.manager.unload_model(model_name)
