"""
Enterprise Vision AI - HuggingFace Space Demo
Gradio-based industrial AI demo with defect detection and ore classification.
"""

import os
import sys

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# Add project root to path so enterprise_vision_ai package is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Try importing YOLO
try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# Try importing utils
try:
    from enterprise_vision_ai.utils.image_utils import preprocess_for_model
    from enterprise_vision_ai.utils.metrics import (
        calculate_anomaly_score,
        calculate_metal_ratio,
        calculate_ore_metrics,
        get_diverter_recommendation,
        get_maintenance_recommendation,
        get_severity_level,
    )
    from enterprise_vision_ai.utils.visualization import (
        draw_annotations,
        get_defect_colors,
        get_ore_class_colors,
    )

    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False

_MODEL_PATH = os.path.join(_ROOT, "yolo11s-seg.pt")
_model_cache: dict = {}


def _get_model():
    """Load YOLO model once, cache it."""
    if "model" not in _model_cache:
        if _YOLO_AVAILABLE and os.path.exists(_MODEL_PATH):
            try:
                _model_cache["model"] = YOLO(_MODEL_PATH)
            except Exception:
                _model_cache["model"] = None
        else:
            _model_cache["model"] = None
    return _model_cache["model"]


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def run_defect_detection(image: Image.Image, conf: float) -> tuple:
    """
    Run defect detection on input image.

    Returns:
        (annotated_image: PIL.Image, results_markdown: str)
    """
    if image is None:
        return None, "Görüntü yüklenmedi."

    model = _get_model()

    if model is None or not _UTILS_AVAILABLE:
        return image, "⚠️ Model yüklenemedi — demo modu. Sonuç üretilemedi."

    image_bgr = _pil_to_bgr(image)
    image_bgr = preprocess_for_model(image_bgr)

    results = model(image_bgr, conf=conf)
    annotated_bgr = draw_annotations(image_bgr, results, get_defect_colors())
    annotated_pil = Image.fromarray(annotated_bgr)  # draw_annotations returns RGB

    score = calculate_anomaly_score(results)
    severity = get_severity_level(score)
    recommendation = get_maintenance_recommendation(score)
    det_count = sum(len(r.boxes) for r in results)

    md = f"""**Tespit Sayısı:** {det_count}
**Anomali Skoru:** {score:.0f}/100
**Severity:** {severity.capitalize()}
**Öneri:** {recommendation}"""

    return annotated_pil, md


def run_ore_classification(image: Image.Image, conf: float) -> tuple:
    """
    Run ore classification on input image.

    Returns:
        (annotated_image: PIL.Image, results_markdown: str)
    """
    if image is None:
        return None, "Görüntü yüklenmedi."

    model = _get_model()

    if model is None or not _UTILS_AVAILABLE:
        return image, "⚠️ Model yüklenemedi — demo modu. Sonuç üretilemedi."

    image_bgr = _pil_to_bgr(image)
    image_bgr = preprocess_for_model(image_bgr)

    results = model(image_bgr, conf=conf)
    annotated_bgr = draw_annotations(image_bgr, results, get_ore_class_colors())
    annotated_pil = Image.fromarray(annotated_bgr)  # draw_annotations returns RGB

    counts = calculate_ore_metrics(results)
    metal_ratio = calculate_metal_ratio(counts)
    diverter = get_diverter_recommendation(metal_ratio)
    total = sum(counts.values())
    dominant = max(counts, key=counts.get) if total > 0 else "—"

    counts_str = " | ".join(f"{k}: {v}" for k, v in counts.items())
    md = f"""**Toplam Tespit:** {total}
**Metal Oranı:** {metal_ratio:.1f}%
**Dominant Sınıf:** {dominant}
**Diverter Önerisi:** {diverter}
**Dağılım:** {counts_str}"""

    return annotated_pil, md


_HEADER_HTML = """
<div style="text-align:center; padding:16px 0 8px;">
  <h1 style="margin:0; font-size:1.8rem;">🏭 Enterprise Vision AI</h1>
  <p style="color:#888; margin:4px 0 0;">Endüstriyel görüntü analizi — defekt tespiti &amp; cevher ön seçimi</p>
</div>
"""

_API_DOCS = """
### 📡 API Kullanımı

Bu HuggingFace Space, Gradio API üzerinden programatik erişime izin verir.

#### Python (gradio_client)
```python
from gradio_client import Client, handle_file

client = Client("https://<kullanici>-<space>.hf.space")

# Defekt tespiti
result = client.predict(
    image=handle_file("my_image.jpg"),
    conf=0.25,
    api_name="/run_defect_detection"
)
annotated_img, results_md = result

# Cevher ön seçimi
result = client.predict(
    image=handle_file("my_ore.jpg"),
    conf=0.25,
    api_name="/run_ore_classification"
)
```

#### curl
```bash
curl -X POST https://<kullanici>-<space>.hf.space/run/predict \\
  -H "Content-Type: application/json" \\
  -d '{"data": ["<base64_image>", 0.25], "fn_index": 0}'
```

`fn_index` değerleri: `0` → Defekt Tespiti, `1` → Cevher Ön Seçimi

> **Not:** API erişimi için Space'in çalışır durumda olması gerekir. `demo.queue()` API erişimini etkinleştirir.
"""

# --- UI ---
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.HTML(_HEADER_HTML)

    with gr.Tabs():
        with gr.Tab("🔍 Defekt Tespiti"):
            gr.Markdown("Yüzey kusurlarını tespit eder, anomali skoru ve bakım önerisi üretir.")
            with gr.Row():
                with gr.Column():
                    defect_img_input = gr.Image(type="pil", label="Görüntü Yükle")
                    defect_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Güven Eşiği")
                    defect_btn = gr.Button("🔍 Analiz Et", variant="primary")
                with gr.Column():
                    defect_img_output = gr.Image(type="pil", label="Sonuç")
                    defect_results_md = gr.Markdown()

            defect_btn.click(
                fn=run_defect_detection,
                inputs=[defect_img_input, defect_conf],
                outputs=[defect_img_output, defect_results_md],
            )

            gr.Examples(
                examples=[["examples/defect_sample.jpg"]],
                inputs=defect_img_input,
                label="Örnek Görüntüler",
            )

        with gr.Tab("💎 Cevher Ön Seçimi"):
            gr.Markdown(
                "Maden cevherlerini sınıflandırır, metal oranı hesaplar ve diverter önerisi üretir."
            )
            with gr.Row():
                with gr.Column():
                    ore_img_input = gr.Image(type="pil", label="Görüntü Yükle")
                    ore_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Güven Eşiği")
                    ore_btn = gr.Button("💎 Analiz Et", variant="primary")
                with gr.Column():
                    ore_img_output = gr.Image(type="pil", label="Sonuç")
                    ore_results_md = gr.Markdown()

            ore_btn.click(
                fn=run_ore_classification,
                inputs=[ore_img_input, ore_conf],
                outputs=[ore_img_output, ore_results_md],
            )

            gr.Examples(
                examples=[["examples/ore_sample.jpg"]],
                inputs=ore_img_input,
                label="Örnek Görüntüler",
            )

    with gr.Accordion("📡 API Kullanımı", open=False):
        gr.Markdown(_API_DOCS)

demo.queue()
demo.launch()
