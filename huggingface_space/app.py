"""
Enterprise Vision AI - HuggingFace Space Demo
Gradio-based web application for industrial AI demos

This Space demonstrates:
1. Defect Detection - Surface defect detection using YOLO
2. Ore Classification - Mineral ore classification and sorting

Author: Enterprise Vision AI Team
License: Apache 2.0
"""

import os
from datetime import datetime
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# Check for torch availability
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using CPU only")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Theme colors matching BAS Industrial AI
THEME = {
    "primary": "#00d4ff",
    "secondary": "#0099cc",
    "background": "#0e1117",
    "surface": "#1a1d23",
    "text": "#ffffff",
    "accent": "#00ff88",
}

# Model configurations - Replace with your HuggingFace model IDs
DEFECT_MODEL_ID = "bas-industriel/yolo-defect-detection"
ORE_MODEL_ID = "bas-industriel/yolo-ore-classification"

# Class names
DEFECT_CLASSES = ["çizik", "çatlak", "delik", "ezilme", "yanık", "pas", "diğer"]

ORE_CLASSES = ["manyetit", "kromit", "pirit", "kalkopirit", "atık", "düşük tenörlü"]

# Global model cache
_defect_model = None
_ore_model = None

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------


def get_device():
    """Get the best available device."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_defect_model():
    """
    Load defect detection model from HuggingFace Hub.
    Uses global cache to avoid reloading.
    """
    global _defect_model

    if _defect_model is not None:
        return _defect_model, True

    try:
        from ultralytics import YOLO

        # Try loading from HuggingFace Hub first
        try:
            model = YOLO(f"huggingface://{DEFECT_MODEL_ID}")
        except Exception as e:
            print(f"Could not load from HF Hub: {e}, using fallback")
            # Fallback to base YOLO model for demo
            model = YOLO("yolo11n-seg.pt")

        _defect_model = model
        return model, True
    except Exception as e:
        print(f"Error loading defect model: {e}")
        return None, False


def load_ore_model():
    """
    Load ore classification model from HuggingFace Hub.
    Uses global cache to avoid reloading.
    """
    global _ore_model

    if _ore_model is not None:
        return _ore_model, True

    try:
        from ultralytics import YOLO

        # Try loading from HuggingFace Hub first
        try:
            model = YOLO(f"huggingface://{ORE_MODEL_ID}")
        except Exception as e:
            print(f"Could not load from HF Hub: {e}, using fallback")
            # Fallback to base YOLO model for demo
            model = YOLO("yolo11n-seg.pt")

        # Set class names for ore classification
        model.names = {i: name for i, name in enumerate(ORE_CLASSES)}

        _ore_model = model
        return model, True
    except Exception as e:
        print(f"Error loading ore model: {e}")
        return None, False


# -----------------------------------------------------------------------------
# INFERENCE FUNCTIONS
# -----------------------------------------------------------------------------


def preprocess_image(image: Image.Image, target_size: int = 640) -> np.ndarray:
    """Preprocess image for model inference."""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize while maintaining aspect ratio
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Calculate scaling
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(img_array, (new_w, new_h))

    # Pad to square
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return padded


def run_defect_detection(image: Image.Image, confidence: float = 0.25):
    """
    Run defect detection on input image.

    Args:
        image: Input PIL Image
        confidence: Confidence threshold for detections

    Returns:
        Annotated image and detection results
    """
    if image is None:
        return None, "Lütfen bir görüntü yükleyiniz."

    try:
        # Load model
        model, loaded = load_defect_model()
        if not loaded or model is None:
            return None, "Model yüklenemedi."

        # Get device
        device = get_device()

        # Run inference
        results = model.predict(image, conf=confidence, verbose=False, device=device)

        # Get result
        result = results[0]

        # Generate annotated image
        annotated_img = result.plot()

        # Parse detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                cls_idx = int(cls)
                cls_name = (
                    DEFECT_CLASSES[cls_idx] if cls_idx < len(DEFECT_CLASSES) else f"sınıf_{cls_idx}"
                )
                detections.append(
                    {"class": cls_name, "confidence": float(conf), "bbox": [float(x) for x in box]}
                )

        # Create results summary
        device_str = "GPU (CUDA)" if device == "cuda" else "CPU"

        summary = f"### 🔍 Algılama Sonuçları\n\n"
        summary += f"- **Toplam Tespit:** {len(detections)}\n"
        summary += f"- **Cihaz:** {device_str}\n\n"

        if detections:
            summary += "#### Tespit Edilen Defectler:\n\n"
            for i, det in enumerate(detections, 1):
                summary += f"{i}. **{det['class']}** - Güven: {det['confidence']:.2%}\n"
        else:
            summary += "Defect tespit edilmedi.\n"

        return Image.fromarray(annotated_img), summary

    except Exception as e:
        return None, f"Hata oluştu: {str(e)}"


def run_ore_classification(image: Image.Image, confidence: float = 0.25):
    """
    Run ore classification on input image.

    Args:
        image: Input PIL Image
        confidence: Confidence threshold for detections

    Returns:
        Annotated image and classification results
    """
    if image is None:
        return None, "Lütfen bir görüntü yükleyiniz."

    try:
        # Load model
        model, loaded = load_ore_model()
        if not loaded or model is None:
            return None, "Model yüklenemedi."

        # Get device
        device = get_device()

        # Run inference
        results = model.predict(image, conf=confidence, verbose=False, device=device)

        # Get result
        result = results[0]

        # Generate annotated image
        annotated_img = result.plot()

        # Parse detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                cls_idx = int(cls)
                cls_name = (
                    ORE_CLASSES[cls_idx] if cls_idx < len(ORE_CLASSES) else f"sınıf_{cls_idx}"
                )
                detections.append(
                    {"class": cls_name, "confidence": float(conf), "bbox": [float(x) for x in box]}
                )

        # Calculate metrics
        class_counts = {}
        for det in detections:
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Create results summary
        device_str = "GPU (CUDA)" if device == "cuda" else "CPU"

        summary = f"### 💎 Sınıflandırma Sonuçları\n\n"
        summary += f"- **Toplam Tespit:** {len(detections)}\n"
        summary += f"- **Cihaz:** {device_str}\n\n"

        if detections:
            summary += "#### Tespit Edilen Cevherler:\n\n"
            for i, det in enumerate(detections, 1):
                summary += f"{i}. **{det['class']}** - Güven: {det['confidence']:.2%}\n"

            summary += "\n#### Özet:\n\n"
            for cls, count in class_counts.items():
                pct = (count / len(detections)) * 100
                summary += f"- {cls}: {count} adet ({pct:.1f}%)\n"
        else:
            summary += "Cevher tespit edilmedi.\n"

        return Image.fromarray(annotated_img), summary

    except Exception as e:
        return None, f"Hata oluştu: {str(e)}"


# -----------------------------------------------------------------------------
# UI COMPONENTS
# -----------------------------------------------------------------------------


def create_header():
    """Create application header."""
    return """
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #0e1117 0%, #1a1d23 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #00d4ff; margin: 0; font-size: 2.5em;">
            🏭 Enterprise Vision AI
        </h1>
        <p style="color: #888; margin: 10px 0 0 0; font-size: 1.2em;">
            Yapay Zeka Destekli Endüstriyel Görüntü Analizi
        </p>
        <div style="margin-top: 15px;">
            <span style="background: #00ff88; color: #000; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                HuggingFace Space Demo
            </span>
        </div>
    </div>
    """


def create_info_card(title: str, description: str, icon: str = "ℹ️"):
    """Create information card."""
    return f"""
    <div style="background: #1a1d23; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #00d4ff;">
        <h3 style="margin: 0 0 10px 0; color: #00d4ff;">{icon} {title}</h3>
        <p style="margin: 0; color: #ccc;">{description}</p>
    </div>
    """


# -----------------------------------------------------------------------------
# GRADIO INTERFACE
# -----------------------------------------------------------------------------


def create_defect_detection_tab():
    """Create defect detection demo tab."""
    with gr.Blocks(title="Defekt Tespiti") as tab:
        gr.HTML(create_header())

        gr.Markdown("## 🔍 Defekt Tespiti")
        gr.HTML(
            create_info_card(
                "Yüzey Kusuru Tespiti",
                "Endüstriyel ürünlerdeki yüzey kusurlarını (çizik, çatlak, delik, pas vb.) "
                "yapay zeka kullanarak tespit eden demo modülü. Görüntüyü yükleyin ve "
                "YOLO tabanlı modelimizin sonuçlarını görün.",
                "🔍",
            )
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="📤 Görüntü Yükle", type="pil", height=400)

                confidence = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Güven Eşiği",
                    info="Düşük değerler daha fazla tespit yapar",
                )

                detect_btn = gr.Button("🔬 Analiz Et", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="📊 Sonuç", type="pil", height=400)

                results_text = gr.Markdown("Sonuçlar burada görünecek", label="Algılama Detayları")

        # Event handlers
        detect_btn.click(
            fn=run_defect_detection,
            inputs=[input_image, confidence],
            outputs=[output_image, results_text],
        )

    return tab


def create_ore_classification_tab():
    """Create ore classification demo tab."""
    with gr.Blocks(title="Cevher Sınıflandırma") as tab:
        gr.HTML(create_header())

        gr.Markdown("## 💎 Cevher Ön Seçimi")
        gr.HTML(
            create_info_card(
                "Maden Cevheri Sınıflandırma",
                "Maden cevherlerini (manyetit, kromit, pirit vb.) sınıflandıran ve "
                "ön seçim yapan yapay zeka sistemi. Madencilik süreçlerinde otomatik "
                "ayrıştırma için kullanılır.",
                "💎",
            )
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="📤 Görüntü Yükle", type="pil", height=400)

                confidence = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Güven Eşiği",
                    info="Düşük değerler daha fazla tespit yapar",
                )

                classify_btn = gr.Button("🔬 Sınıflandır", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="📊 Sonuç", type="pil", height=400)

                results_text = gr.Markdown(
                    "Sonuçlar burada görünecek", label="Sınıflandırma Detayları"
                )

        # Event handlers
        classify_btn.click(
            fn=run_ore_classification,
            inputs=[input_image, confidence],
            outputs=[output_image, results_text],
        )

    return tab


def create_about_tab():
    """Create about/info tab."""
    with gr.Blocks(title="Hakkında") as tab:
        gr.HTML(create_header())

        gr.Markdown("## 📋 Proje Hakkında")

        gr.Markdown("""
        ### Enterprise Vision AI
        
        Endüstriyel yapay zeka çözümleri sunan bir platformdur. 
        Ana fonksiyonları:
        
        - **Defekt Tespiti**: Ürün yüzeylerindeki kusurların otomatik tespiti
        - **Cevher Ön Seçimi**: Madencilik süreçlerinde cevher sınıflandırma
        
        ### Teknoloji
        
        - YOLO (You Only Look Once) nesne tespit algoritması
        - Ultralytics framework
        - PyTorch backend
        - Gradio web arayüzü
        
        ### HuggingFace Space
        
        Bu demo, HuggingFace Spaces platformunda barındırılmaktadır.
        Modeller HuggingFace Hub'dan yüklenmektedir.
        
        ---
        
        **Not**: Bu bir demo uygulamasıdır. Gerçek model ağırlıkları HuggingFace Hub'a 
        yüklendikten sonra otomatik olarak kullanılacaktır.
        """)

        gr.Markdown("""
        ### 🤗 HuggingFace Hub Modelleri
        
        Bu Space'de kullanılan modeller:
        
        - [Defect Detection Model](https://huggingface.co/bas-industriel/yolo-defect-detection)
        - [Ore Classification Model](https://huggingface.co/bas-industriel/yolo-ore-classification)
        
        ---
        
        *© 2024 Enterprise Vision AI - Tüm hakları saklıdır*
        """)

    return tab


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------


def create_app():
    """Create main Gradio application."""

    # Custom CSS
    custom_css = """
    /* Theme Colors */
    :root {
        --primary: #00d4ff;
        --secondary: #0099cc;
        --accent: #00ff88;
        --background: #0e1117;
        --surface: #1a1d23;
    }
    
    /* Button styles */
    .primary-btn {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    /* Card styles */
    .card {
        background: #1a1d23;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Header gradient */
    .header {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d23 100%);
    }
    """

    # Create main interface
    with gr.Blocks(title="Enterprise Vision AI", css=custom_css) as app:

        # Create tabs
        with gr.TabbedInterface(
            tab_names=["🔍 Defekt Tespiti", "💎 Cevher Sınıflandırma", "ℹ️ Hakkında"],
            tab_creates=[
                create_defect_detection_tab,
                create_ore_classification_tab,
                create_about_tab,
            ],
        ) as tabs:
            pass  # Tabs are created by the tabbed_interface

    return app


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Create and launch app
    app = create_app()

    # Launch with appropriate settings for HuggingFace Spaces
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
