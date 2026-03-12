# FastAPI Fix & Run Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan.

**Goal:** Fix all broken imports in the FastAPI layer, create missing services, and run both FastAPI (port 8000) and Streamlit (port 8501) together.

**Architecture:** The FastAPI code was written with bare module imports (`from api.routes import ...`) that only work if run from a specific directory. Fix by making all imports package-qualified (`from enterprise_vision_ai.api.routes import ...`). Two missing services (`InferenceService`, `ModelService`) need to be created in `src/enterprise_vision_ai/services/`.

**Tech Stack:** FastAPI, uvicorn, Streamlit, ultralytics YOLO, uv

---

## Files to Change

| File | Action | Issue |
|------|--------|-------|
| `src/enterprise_vision_ai/api/main.py` | Modify | 3 bare imports |
| `src/enterprise_vision_ai/api/dependencies.py` | Modify | 3 bare imports |
| `src/enterprise_vision_ai/api/routes/inference.py` | Modify | 5 bare imports |
| `src/enterprise_vision_ai/api/routes/upload.py` | Modify | 1 bare import |
| `src/enterprise_vision_ai/services/__init__.py` | Create | new package |
| `src/enterprise_vision_ai/services/inference_service.py` | Create | was deleted |
| `src/enterprise_vision_ai/services/model_service.py` | Create | was deleted |

---

## Chunk 1: Fix Import Paths

### Task 1: Fix api/main.py

**Files:** Modify `src/enterprise_vision_ai/api/main.py`

- [ ] **Step 1:** Fix 3 bare imports

```python
# OLD:
from api.routes import datasets, health, inference, models, upload
from api.schemas.response import HealthResponse
# in lifespan:
from clients.yolo_client import yolo_model_manager
# in health_check:
from clients.yolo_client import yolo_model_manager

# NEW:
from enterprise_vision_ai.api.routes import datasets, health, inference, models, upload
from enterprise_vision_ai.api.schemas.response import HealthResponse
# in lifespan and health_check:
from enterprise_vision_ai.clients.yolo_client import yolo_model_manager
```

- [ ] **Step 2:** Verify import works
```bash
uv run python -c "from enterprise_vision_ai.api import main; print('main OK')"
```

---

### Task 2: Fix api/dependencies.py

**Files:** Modify `src/enterprise_vision_ai/api/dependencies.py`

- [ ] **Step 1:** Fix imports (services don't exist yet — will be fixed in Chunk 2)

```python
# OLD:
from clients.yolo_client import YOLOModelManager
from services.inference_service import InferenceService
from services.model_service import ModelService

# NEW:
from enterprise_vision_ai.clients.yolo_client import YOLOModelManager
from enterprise_vision_ai.services.inference_service import InferenceService
from enterprise_vision_ai.services.model_service import ModelService
```

---

### Task 3: Fix api/routes/inference.py

**Files:** Modify `src/enterprise_vision_ai/api/routes/inference.py`

- [ ] **Step 1:** Fix 5 bare imports

```python
# OLD:
from api.dependencies import get_inference_service, get_model_service
from api.schemas.request import InferenceRequest, ModelListRequest
from api.schemas.response import (ClassificationResult, DetectionResult,
    ErrorResponse, InferenceResponse, ModelInfo)
from services.inference_service import InferenceService
from services.model_service import ModelService

# NEW:
from enterprise_vision_ai.api.dependencies import get_inference_service, get_model_service
from enterprise_vision_ai.api.schemas.request import InferenceRequest, ModelListRequest
from enterprise_vision_ai.api.schemas.response import (ClassificationResult, DetectionResult,
    ErrorResponse, InferenceResponse, ModelInfo)
from enterprise_vision_ai.services.inference_service import InferenceService
from enterprise_vision_ai.services.model_service import ModelService
```

---

### Task 4: Fix api/routes/upload.py

**Files:** Modify `src/enterprise_vision_ai/api/routes/upload.py`

- [ ] **Step 1:** Fix 1 bare import

```python
# OLD:
from api.schemas.response import BatchUploadResponse, ErrorResponse, UploadedFile, UploadResponse

# NEW:
from enterprise_vision_ai.api.schemas.response import BatchUploadResponse, ErrorResponse, UploadedFile, UploadResponse
```

---

## Chunk 2: Create Missing Services

### Task 5: Create InferenceService

**Files:** Create `src/enterprise_vision_ai/services/inference_service.py`

```python
"""
Inference Service - YOLO inference for FastAPI endpoints.
"""
import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from enterprise_vision_ai.clients.yolo_client import YOLOModelManager

DEFAULT_MODEL = "yolo11n-seg.pt"


class InferenceService:
    def __init__(self, manager: YOLOModelManager):
        self.manager = manager

    async def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)

    async def _load_image_from_data(self, image_data: str) -> np.ndarray:
        # base64 encoded string
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        return await self._load_image_from_bytes(image_bytes)

    def _results_to_dict(self, results: Any, model_name: str) -> Dict:
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
        image = await self._load_image_from_data(image_data)
        model = await self.manager.get_model(model_path)
        results = model.predict(source=image, conf=confidence_threshold, verbose=False)
        return self._results_to_dict(results, model_path)

    async def detect_defects_from_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
    ) -> Dict:
        model_path = model_name or DEFAULT_MODEL
        image = await self._load_image_from_bytes(image_bytes)
        model = await self.manager.get_model(model_path)
        results = model.predict(source=image, conf=confidence_threshold, verbose=False)
        return self._results_to_dict(results, model_path)

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
```

---

### Task 6: Create ModelService

**Files:** Create `src/enterprise_vision_ai/services/model_service.py`

```python
"""
Model Service - Model management for FastAPI endpoints.
"""
from typing import Any, Dict, List, Optional

from enterprise_vision_ai.clients.yolo_client import YOLOModelManager

AVAILABLE_MODELS = [
    {"name": "yolo11n-seg.pt", "version": "11n", "type": "segmentation"},
    {"name": "yolo11s-seg.pt", "version": "11s", "type": "segmentation"},
    {"name": "yolo11n.pt",     "version": "11n", "type": "detection"},
]


class ModelService:
    def __init__(self, manager: YOLOModelManager):
        self.manager = manager

    async def list_models(self) -> List[Dict]:
        loaded = self.manager.get_loaded_models()
        return [
            {
                "name": m["name"],
                "version": m["version"],
                "type": m["type"],
                "loaded": m["name"] in loaded,
                "description": f"YOLO {m['version']} {m['type']} model",
            }
            for m in AVAILABLE_MODELS
        ]

    async def load_model(self, model_name: str) -> bool:
        return await self.manager.load_model(model_name)

    async def unload_model(self, model_name: str) -> bool:
        return await self.manager.unload_model(model_name)
```

---

### Task 7: Create services __init__.py

**Files:** Create `src/enterprise_vision_ai/services/__init__.py`

```python
"""Enterprise Vision AI Services."""
from enterprise_vision_ai.services.inference_service import InferenceService
from enterprise_vision_ai.services.model_service import ModelService

__all__ = ["InferenceService", "ModelService"]
```

---

## Chunk 3: Verify and Run

### Task 8: Verify API imports

```bash
uv run python -c "
from enterprise_vision_ai.api.main import app
print('FastAPI app import OK')
print('Routes:', [r.path for r in app.routes[:5]])
"
```

Expected: `FastAPI app import OK`

### Task 9: Run FastAPI backend

```bash
uv run uvicorn enterprise_vision_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Task 10: Run Streamlit frontend (second terminal)

```bash
uv run streamlit run app.py --server.port 8501
```

### Task 11: Commit

```bash
git add src/enterprise_vision_ai/api/ src/enterprise_vision_ai/services/
git commit -m "fix: repair FastAPI imports, add InferenceService and ModelService"
```
