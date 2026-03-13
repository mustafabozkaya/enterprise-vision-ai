"""
Enterprise Vision AI - HuggingFace Space Demo
Standalone Gradio app — no external package dependencies.
Model: Mustafaege/enterprise-vision-ai-models (yolo11s-seg.pt)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# Model loading (from HF Hub)
# ---------------------------------------------------------------------------
_HF_MODEL_REPO = "Mustafaege/enterprise-vision-ai-models"
_MODEL_FILENAME = "yolo11s-seg.pt"
_model_cache: dict = {}


def _get_model():
    """Download yolo11s-seg.pt from HF Hub (cached) and return YOLO instance."""
    if "model" in _model_cache:
        return _model_cache["model"]
    try:
        from ultralytics import YOLO

        path = hf_hub_download(repo_id=_HF_MODEL_REPO, filename=_MODEL_FILENAME)
        _model_cache["model"] = YOLO(path)
        print(f"Model loaded from {path}")
    except Exception as exc:
        print(f"Model could not be loaded: {exc}")
        _model_cache["model"] = None
    return _model_cache["model"]


# Pre-load model at startup so first inference is fast
_get_model()

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
_DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "çatlak": (255, 0, 0),
    "çizik": (0, 255, 255),
    "delik": (255, 165, 0),
    "leke": (128, 0, 128),
    "deformasyon": (0, 0, 255),
}
_ORE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "manyetit": (220, 20, 60),
    "krom": (0, 255, 127),
    "atık": (128, 128, 128),
    "düşük tenör": (255, 165, 0),
}
_DEFAULT_COLOR = (255, 255, 255)

# ---------------------------------------------------------------------------
# Inline utilities (no enterprise_vision_ai dependency)
# ---------------------------------------------------------------------------


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _draw_annotations(
    image_bgr: np.ndarray,
    results,
    colors: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    """Draw segmentation masks + bounding boxes on image. Returns RGB ndarray."""
    out = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if not results:
        return out
    for r in results:
        # Segmentation masks
        if getattr(r, "masks", None) is not None and r.boxes is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.data.cpu().numpy()
            h, w = out.shape[:2]
            for mask, box in zip(masks, boxes):
                cls_id = int(box[5])
                name = r.names.get(cls_id, f"cls_{cls_id}") if r.names else f"cls_{cls_id}"
                color = colors.get(name, _DEFAULT_COLOR)
                mask_u8 = (mask * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)
                overlay = np.zeros_like(out)
                for i, c in enumerate(color):
                    overlay[:, :, i] = mask_resized * (c / 255.0)
                out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)
        # Bounding boxes + labels
        if getattr(r, "boxes", None) is not None:
            for box in r.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                name = r.names.get(cls_id, f"cls_{cls_id}") if r.names else f"cls_{cls_id}"
                color = colors.get(name, _DEFAULT_COLOR)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
                cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def _anomaly_score(results) -> float:
    if not results:
        return 0.0
    total, count = 0.0, 0
    for r in results:
        if getattr(r, "boxes", None) is not None:
            for box in r.boxes.data.cpu().numpy():
                total += float(box[4]) * 100
                count += 1
    return min(100.0, total / count) if count else 0.0


def _severity(score: float) -> str:
    if score < 30:
        return "düşük"
    if score < 70:
        return "orta"
    return "yüksek"


def _maintenance(score: float) -> str:
    if score < 30:
        return "✅ Normal. Rutin bakım yeterli."
    if score < 50:
        return "⚠️ Hafif anomali. Yakın takip önerilir."
    if score < 70:
        return "🔶 Orta anomali. Planlı bakım önerilir."
    if score < 85:
        return "🔴 Yüksek anomali. Acil bakım gerekli."
    return "🚨 Kritik! Üretim durdurulmalı."


def _ore_counts(results) -> Dict[str, int]:
    counts = {"manyetit": 0, "krom": 0, "atık": 0, "düşük tenör": 0}
    if not results:
        return counts
    for r in results:
        if getattr(r, "boxes", None) is not None:
            for box in r.boxes.data.cpu().numpy():
                cls_id = int(box[5])
                name = (r.names.get(cls_id, "") if r.names else "").lower()
                for key in counts:
                    if key in name:
                        counts[key] += 1
                        break
    return counts


def _metal_ratio(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts.get("manyetit", 0) + counts.get("krom", 0)) / total * 100


def _diverter(ratio: float) -> str:
    if ratio >= 60:
        return "⬆️ İLERİ İŞLEM: Yüksek metal oranı — üretime gönder."
    if ratio >= 30:
        return "⏸️ KONTROL: Orta metal oranı — ek analiz gerekli."
    return "⬇️ ATIK: Düşük metal oranı — atık hattına yönlendir."


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------


def run_defect_detection(image: Image.Image, conf: float) -> tuple:
    if image is None:
        return None, "⚠️ Görüntü yüklenmedi."

    model = _get_model()
    if model is None:
        return image, "❌ Model yüklenemedi. Lütfen daha sonra tekrar deneyin."

    image_bgr = _pil_to_bgr(image)
    results = model(image_bgr, conf=conf, verbose=False)

    annotated = _draw_annotations(image_bgr, results, _DEFECT_COLORS)
    score = _anomaly_score(results)
    severity = _severity(score)
    recommendation = _maintenance(score)
    det_count = sum(len(r.boxes) for r in results if getattr(r, "boxes", None) is not None)

    md = (
        f"**Tespit Sayısı:** {det_count}  \n"
        f"**Anomali Skoru:** {score:.0f} / 100  \n"
        f"**Severity:** {severity.capitalize()}  \n"
        f"**Öneri:** {recommendation}"
    )
    return Image.fromarray(annotated), md


def run_ore_classification(image: Image.Image, conf: float) -> tuple:
    if image is None:
        return None, "⚠️ Görüntü yüklenmedi."

    model = _get_model()
    if model is None:
        return image, "❌ Model yüklenemedi. Lütfen daha sonra tekrar deneyin."

    image_bgr = _pil_to_bgr(image)
    results = model(image_bgr, conf=conf, verbose=False)

    annotated = _draw_annotations(image_bgr, results, _ORE_COLORS)
    counts = _ore_counts(results)
    ratio = _metal_ratio(counts)
    diverter = _diverter(ratio)
    total = sum(counts.values())
    dominant = max(counts, key=counts.get) if total > 0 else "—"
    dist = " | ".join(f"{k}: {v}" for k, v in counts.items())

    md = (
        f"**Toplam Tespit:** {total}  \n"
        f"**Metal Oranı:** {ratio:.1f}%  \n"
        f"**Dominant Sınıf:** {dominant}  \n"
        f"**Diverter Önerisi:** {diverter}  \n"
        f"**Dağılım:** {dist}"
    )
    return Image.fromarray(annotated), md


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_HEADER = """
<div style="text-align:center;padding:16px 0 8px;">
  <h1 style="margin:0;font-size:1.8rem;">🏭 Enterprise Vision AI</h1>
  <p style="color:#888;margin:4px 0 0;">
    Endüstriyel görüntü analizi — defekt tespiti &amp; cevher ön seçimi
  </p>
</div>
"""

_API_DOCS = """
### 📡 API Kullanımı

```python
from gradio_client import Client, handle_file

client = Client("Mustafaege/enterprise-vision-ai")

# Defekt tespiti
annotated_img, results_md = client.predict(
    image=handle_file("surface.jpg"),
    conf=0.25,
    api_name="/run_defect_detection",
)

# Cevher ön seçimi
annotated_img, results_md = client.predict(
    image=handle_file("ore.jpg"),
    conf=0.25,
    api_name="/run_ore_classification",
)
```
"""

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.HTML(_HEADER)

    with gr.Tabs():
        # ── Tab 1: Defect Detection ──────────────────────────────────────
        with gr.Tab("🔍 Defekt Tespiti"):
            gr.Markdown("Yüzey kusurlarını tespit eder, anomali skoru ve bakım önerisi üretir.")
            with gr.Row():
                with gr.Column():
                    defect_img_in = gr.Image(type="pil", label="Görüntü Yükle")
                    defect_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Güven Eşiği")
                    defect_btn = gr.Button("🔍 Analiz Et", variant="primary")
                with gr.Column():
                    defect_img_out = gr.Image(type="pil", label="Sonuç")
                    defect_md = gr.Markdown()

            defect_btn.click(
                fn=run_defect_detection,
                inputs=[defect_img_in, defect_conf],
                outputs=[defect_img_out, defect_md],
            )
            gr.Examples(
                examples=[["examples/defect_sample.jpg", 0.25]],
                inputs=[defect_img_in, defect_conf],
                label="Örnek",
            )

        # ── Tab 2: Ore Classification ─────────────────────────────────────
        with gr.Tab("💎 Cevher Ön Seçimi"):
            gr.Markdown(
                "Maden cevherlerini sınıflandırır, metal oranı hesaplar ve diverter önerisi üretir."
            )
            with gr.Row():
                with gr.Column():
                    ore_img_in = gr.Image(type="pil", label="Görüntü Yükle")
                    ore_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Güven Eşiği")
                    ore_btn = gr.Button("💎 Analiz Et", variant="primary")
                with gr.Column():
                    ore_img_out = gr.Image(type="pil", label="Sonuç")
                    ore_md = gr.Markdown()

            ore_btn.click(
                fn=run_ore_classification,
                inputs=[ore_img_in, ore_conf],
                outputs=[ore_img_out, ore_md],
            )
            gr.Examples(
                examples=[["examples/ore_sample.jpg", 0.25]],
                inputs=[ore_img_in, ore_conf],
                label="Örnek",
            )

    with gr.Accordion("📡 API Kullanımı", open=False):
        gr.Markdown(_API_DOCS)

demo.queue()
demo.launch()
