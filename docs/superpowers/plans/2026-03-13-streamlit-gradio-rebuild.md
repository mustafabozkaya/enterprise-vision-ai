# Streamlit + Gradio UI Rebuild Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Streamlit UI (app.py + defect.py + ore.py) and Gradio HF Space (huggingface_space/app.py) from scratch per spec.

**Architecture:** Full-page rewrite approach — each file replaced entirely. Streamlit uses shared sidebar from `pages/_sidebar.py`. Inference imports come from `enterprise_vision_ai.utils.*` directly (not `services/utils.py` which is deprecated). Gradio uses `gr.Blocks` + `gr.Tabs` pattern.

**Tech Stack:** Streamlit, Gradio, YOLO (ultralytics), enterprise_vision_ai.utils (image_utils, visualization, metrics), numpy, PIL, cv2

**Spec:** `docs/superpowers/specs/2026-03-13-streamlit-gradio-redesign.md`

---

## Chunk 1: Streamlit Sayfaları

### Task 1: Rewrite app.py (Ana Sayfa)

**Files:**
- Modify: `app.py` (full rewrite)

- [ ] **Step 1: Rewrite app.py**

Replace entire contents with:

```python
"""
Enterprise Vision AI - Ana Sayfa
Demo/sunum odaklı bilgi sayfası
"""

import streamlit as st
from pages._sidebar import render_sidebar

st.set_page_config(
    page_title="Enterprise Vision AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar(active_page="")

# --- Hero ---
st.title("🏭 Enterprise Vision AI")
st.markdown("Endüstriyel görüntü analizi — gerçek zamanlı defekt tespiti ve cevher ön seçimi.")

st.divider()

# --- Metrik satırı ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Aktif Modüller", "2")
m2.metric("Model Doğruluğu", "%94")
m3.metric("İşlem Hızı", "30 FPS")
m4.metric("Desteklenen Sınıf", "9+")

st.divider()

# --- Modül kartları ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="border:1px solid #30363d; border-radius:8px; padding:20px; min-height:200px;">
        <h3>🔍 Defekt Tespiti</h3>
        <ul>
        <li>Yüzey kusurlarını tespit eder</li>
        <li>Anomali skoru hesaplar (0–100)</li>
        <li>Severity ve bakım önerisi üretir</li>
        <li>Segmentasyon maskeleri çizer</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/01_Defekt_Tespiti.py", label="Defekt Tespiti Modülüne Git →", icon="🔍")

with col2:
    st.markdown(
        """
        <div style="border:1px solid #30363d; border-radius:8px; padding:20px; min-height:200px;">
        <h3>💎 Cevher Ön Seçimi</h3>
        <ul>
        <li>Maden cevherlerini sınıflandırır</li>
        <li>Metal oranı hesaplar (%)</li>
        <li>Diverter yönlendirme önerisi üretir</li>
        <li>Sınıf dağılım grafiği gösterir</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/02_Cevher_On_Secimi.py", label="Cevher Seçimi Modülüne Git →", icon="💎")

st.divider()

# --- Nasıl Kullanılır ---
with st.expander("📖 Defekt Tespiti — Nasıl Kullanılır?"):
    st.markdown(
        """
        1. Sol menüden **Defekt Tespiti** sayfasına gidin
        2. JPG / JPEG / PNG formatında görüntü yükleyin
        3. Güven eşiğini sidebar'dan ayarlayın (varsayılan: %25)
        4. **Analiz Et** butonuna tıklayın
        5. Annotated görüntü, metrikler ve tespit tablosu görünecektir
        """
    )

with st.expander("📖 Cevher Ön Seçimi — Nasıl Kullanılır?"):
    st.markdown(
        """
        1. Sol menüden **Cevher Ön Seçimi** sayfasına gidin
        2. Cevher görüntüsü yükleyin (JPG, JPEG, PNG)
        3. Güven eşiğini sidebar'dan ayarlayın
        4. **Analiz Et** butonuna tıklayın
        5. Metal oranı, diverter önerisi ve sınıf dağılım grafiği görünecektir
        """
    )

# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align:center; color:#666; font-size:12px;'>© 2026 Enterprise Vision AI</div>",
    unsafe_allow_html=True,
)
```

- [ ] **Step 2: Run existing tests — confirm nothing broken**

```bash
uv run pytest tests/ -q
```

Expected: 29 passed, 0 failed

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: rebuild app.py as info/showcase page with shared sidebar"
```

---

### Task 2: Rewrite pages/defect.py (Defekt Tespiti Modülü)

**Files:**
- Modify: `pages/defect.py` (full rewrite)

Key facts:
- Wrapper `pages/01_Defekt_Tespiti.py` calls `from pages.defect import render` then `render()` — `render()` function is mandatory
- Model file at project root: `yolo11s-seg.pt`
- Imports: `enterprise_vision_ai.utils.image_utils`, `enterprise_vision_ai.utils.visualization`, `enterprise_vision_ai.utils.metrics`
- `load_image(file)` → `np.ndarray` (BGR), `preprocess_for_model(image)` → `np.ndarray`
- `draw_annotations(image, results, class_colors)` → `np.ndarray` (RGB output)
- `calculate_anomaly_score(results)` → float 0–100
- `get_severity_level(score)` → "düşük" | "orta" | "yüksek"
- `get_maintenance_recommendation(score)` → str
- `create_metrics_dataframe(data: List[Dict])` → DataFrame (columns = dict keys)
- `get_defect_colors()` → Dict[str, Tuple[int,int,int]]

- [ ] **Step 1: Rewrite pages/defect.py**

Replace entire contents with:

```python
"""
Enterprise Vision AI - Defekt Tespiti Modülü
Yüzey kusurlarının tespiti ve analizi
"""

import cv2
import numpy as np
import streamlit as st

from enterprise_vision_ai.utils.image_utils import load_image, preprocess_for_model
from enterprise_vision_ai.utils.metrics import (
    calculate_anomaly_score,
    create_metrics_dataframe,
    get_maintenance_recommendation,
    get_severity_level,
)
from enterprise_vision_ai.utils.visualization import draw_annotations, get_defect_colors
from pages._sidebar import render_sidebar

MODEL_PATH = "yolo11s-seg.pt"
DEFECT_CLASSES = ["çatlak", "çizik", "delik", "leke", "deformasyon"]


@st.cache_resource
def _load_model():
    """YOLO modelini yükle. Başarısız olursa None döner (demo modu)."""
    try:
        from ultralytics import YOLO

        return YOLO(MODEL_PATH)
    except Exception:
        return None


def render():
    # Note: set_page_config is NOT called here.
    # Sub-pages in Streamlit MPA inherit config from app.py.
    # Calling set_page_config inside render() imported by a wrapper
    # can raise StreamlitAPIException in some Streamlit versions.

    # --- Sidebar ---
    render_sidebar(active_page="defect")
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.25, 0.05)
        show_masks = st.checkbox("Segmentasyon Maskeleri", value=True)
        show_boxes = st.checkbox("Bounding Box", value=True)

        model = _load_model()
        status_color = "#238636" if model is not None else "#da3633"
        status_text = "Yüklendi" if model is not None else "Demo Modu"
        st.markdown(
            f'<div style="border-left:3px solid {status_color}; padding:8px 12px;'
            f' margin-top:8px; border-radius:4px; background:#161b22;">'
            f"<strong>Model Durumu</strong><br>"
            f'<span style="color:{status_color};">● {status_text}</span></div>',
            unsafe_allow_html=True,
        )

    # --- Başlık ---
    st.title("🔍 Defekt Tespiti")
    st.markdown("Endüstriyel yüzey kusurlarını tespit eder ve anomali skoru hesaplar.")

    # --- Yükleme ---
    uploaded_file = st.file_uploader(
        "Görüntü Yükle", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    # --- Analiz Et butonu ---
    _, btn_col, _ = st.columns([1, 2, 1])
    analyze_clicked = btn_col.button(
        "🔍 Analiz Et", type="primary", use_container_width=True
    )

    # --- Boş durum ---
    if uploaded_file is None:
        st.info("JPG, JPEG veya PNG formatında bir görüntü yükleyin.")
        st.markdown("**Tespit Edilen Sınıflar:** " + " • ".join(DEFECT_CLASSES))
        return

    # --- Inference (buton tıklandığında) ---
    if analyze_clicked:
        image_bgr = load_image(uploaded_file)
        if image_bgr is None:
            st.error("Görüntü yüklenemedi. Lütfen geçerli bir dosya seçin.")
            return

        image_bgr = preprocess_for_model(image_bgr)
        model = _load_model()

        if model is not None:
            results = model(image_bgr, conf=confidence)
            annotated = draw_annotations(image_bgr, results, get_defect_colors())
        else:
            results = None
            annotated = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            st.warning("⚠️ Model yüklenemedi — demo modu. Orijinal görüntü gösteriliyor.")

        score = calculate_anomaly_score(results) if results else 0.0
        severity = get_severity_level(score)
        recommendation = get_maintenance_recommendation(score)
        det_count = sum(len(r.boxes) for r in results) if results else 0

        rows = []
        if results:
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf_val, cls_id = box
                        class_name = r.names.get(int(cls_id), f"cls_{int(cls_id)}")
                        rows.append(
                            {
                                "Sınıf": class_name,
                                "Güven": f"{conf_val:.2f}",
                                "Bbox": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                            }
                        )

        st.session_state["defect_result"] = {
            "img": annotated,
            "score": score,
            "severity": severity,
            "recommendation": recommendation,
            "count": det_count,
            "rows": rows,
        }

    # --- Sonuçları göster ---
    if "defect_result" in st.session_state:
        res = st.session_state["defect_result"]

        st.image(res["img"], use_column_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tespit Sayısı", res["count"])
        c2.metric("Anomali Skoru", f"{res['score']:.0f}/100")
        c3.metric("Severity", res["severity"].capitalize())
        if res["score"] >= 85:
            oneri = "Acil"
        elif res["score"] >= 30:
            oneri = "Yakın Bakım"
        else:
            oneri = "Rutin"
        c4.metric("Öneri", oneri)

        if res["rows"]:
            st.dataframe(
                create_metrics_dataframe(res["rows"]), use_container_width=True
            )
        else:
            st.info("Bu görüntüde tespit bulunamadı.")
```

- [ ] **Step 2: Run existing tests**

```bash
uv run pytest tests/ -q
```

Expected: 29 passed, 0 failed

- [ ] **Step 3: Commit**

```bash
git add pages/defect.py
git commit -m "feat: rebuild defect.py with button-triggered inference and top/bottom layout"
```

---

### Task 3: Rewrite pages/ore.py (Cevher Ön Seçimi Modülü)

**Files:**
- Modify: `pages/ore.py` (full rewrite)

Key facts:
- Wrapper `pages/02_Cevher_On_Secimi.py` calls `from pages.ore import render` then `render()` — `render()` function is mandatory
- `calculate_ore_metrics(results)` → `{"manyetit": int, "krom": int, "atık": int, "düşük tenör": int}`
- `calculate_metal_ratio(counts)` → float 0–100
- `get_diverter_recommendation(metal_ratio)` → str
- `create_ore_dataframe(counts)` → DataFrame with columns ["Class", "Count"]
- `get_ore_class_colors()` → Dict[str, Tuple[int,int,int]]
- Pie chart: removed per spec. Only bar chart.

- [ ] **Step 1: Rewrite pages/ore.py**

Replace entire contents with:

```python
"""
Enterprise Vision AI - Cevher Ön Seçimi Modülü
Maden cevherlerinin sınıflandırılması ve ayrıştırılması
"""

import cv2
import numpy as np
import streamlit as st

from enterprise_vision_ai.utils.image_utils import load_image, preprocess_for_model
from enterprise_vision_ai.utils.metrics import (
    calculate_metal_ratio,
    calculate_ore_metrics,
    create_ore_dataframe,
    get_diverter_recommendation,
)
from enterprise_vision_ai.utils.visualization import draw_annotations, get_ore_class_colors
from pages._sidebar import render_sidebar

MODEL_PATH = "yolo11s-seg.pt"
ORE_CLASSES = {
    "manyetit": "Demir içerikli manyetit cevheri",
    "krom": "Krom cevheri",
    "atık": "Değersiz atık materyal",
    "düşük tenör": "Düşük metal içerikli cevher",
}


@st.cache_resource
def _load_model():
    """YOLO modelini yükle. Başarısız olursa None döner (demo modu)."""
    try:
        from ultralytics import YOLO

        return YOLO(MODEL_PATH)
    except Exception:
        return None


def render():
    # Note: set_page_config is NOT called here (same reason as defect.py).

    # --- Sidebar ---
    render_sidebar(active_page="ore")
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.25, 0.05)
        show_masks = st.checkbox("Segmentasyon Maskeleri", value=True)
        show_boxes = st.checkbox("Bounding Box", value=True)

        model = _load_model()
        status_color = "#238636" if model is not None else "#da3633"
        status_text = "Yüklendi" if model is not None else "Demo Modu"
        st.markdown(
            f'<div style="border-left:3px solid {status_color}; padding:8px 12px;'
            f' margin-top:8px; border-radius:4px; background:#161b22;">'
            f"<strong>Model Durumu</strong><br>"
            f'<span style="color:{status_color};">● {status_text}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### 🗂️ Sınıflar")
        classes_html = "".join(
            f'<div style="padding:4px 0; border-bottom:1px solid #21262d;">'
            f"<strong>{cls}</strong><br>"
            f'<span style="color:#8b949e; font-size:12px;">{desc}</span></div>'
            for cls, desc in ORE_CLASSES.items()
        )
        st.markdown(
            f'<div style="border-left:3px solid #1f6feb; padding:8px 12px;'
            f' border-radius:4px; background:#161b22;">{classes_html}</div>',
            unsafe_allow_html=True,
        )

    # --- Başlık ---
    st.title("💎 Cevher Ön Seçimi")
    st.markdown(
        "Maden cevherlerini sınıflandırır, metal oranı hesaplar ve diverter yönü önerir."
    )

    # --- Yükleme ---
    uploaded_file = st.file_uploader(
        "Görüntü Yükle", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    # --- Analiz Et butonu ---
    _, btn_col, _ = st.columns([1, 2, 1])
    analyze_clicked = btn_col.button(
        "💎 Analiz Et", type="primary", use_container_width=True
    )

    # --- Boş durum ---
    if uploaded_file is None:
        st.info("Analiz için bir cevher görüntüsü yükleyin (JPG, JPEG, PNG).")
        st.markdown(
            "**Metal Oranı Formülü:** `(manyetit + krom) / toplam_tespit × 100`"
        )
        return

    # --- Inference (buton tıklandığında) ---
    if analyze_clicked:
        image_bgr = load_image(uploaded_file)
        if image_bgr is None:
            st.error("Görüntü yüklenemedi. Lütfen geçerli bir dosya seçin.")
            return

        image_bgr = preprocess_for_model(image_bgr)
        model = _load_model()

        if model is not None:
            results = model(image_bgr, conf=confidence)
            annotated = draw_annotations(image_bgr, results, get_ore_class_colors())
        else:
            results = None
            annotated = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            st.warning("⚠️ Model yüklenemedi — demo modu. Orijinal görüntü gösteriliyor.")

        counts = calculate_ore_metrics(results) if results else {
            "manyetit": 0, "krom": 0, "atık": 0, "düşük tenör": 0
        }
        metal_ratio = calculate_metal_ratio(counts)
        diverter = get_diverter_recommendation(metal_ratio)
        total_count = sum(counts.values())
        dominant = max(counts, key=counts.get) if total_count > 0 else "—"

        st.session_state["ore_result"] = {
            "img": annotated,
            "counts": counts,
            "metal_ratio": metal_ratio,
            "diverter": diverter,
            "total": total_count,
            "dominant": dominant,
        }

    # --- Sonuçları göster ---
    if "ore_result" in st.session_state:
        res = st.session_state["ore_result"]

        st.image(res["img"], use_column_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Tespit", res["total"])
        c2.metric("Metal Oranı", f"{res['metal_ratio']:.1f}%")
        if res["metal_ratio"] >= 60:
            diverter_short = "İleri"
        elif res["metal_ratio"] >= 30:
            diverter_short = "Kontrol"
        else:
            diverter_short = "Atık"
        c3.metric("Diverter Önerisi", diverter_short)
        c4.metric("Dominant Sınıf", res["dominant"].capitalize())

        df = create_ore_dataframe(res["counts"])
        st.bar_chart(df.set_index("Class")["Count"])
```

- [ ] **Step 2: Run existing tests**

```bash
uv run pytest tests/ -q
```

Expected: 29 passed, 0 failed

- [ ] **Step 3: Commit**

```bash
git add pages/ore.py
git commit -m "feat: rebuild ore.py with button-triggered inference, bar chart, top/bottom layout"
```

---

## Chunk 2: Gradio HF Space

### Task 4: Create example placeholder images

**Files:**
- Create: `huggingface_space/examples/defect_sample.jpg`
- Create: `huggingface_space/examples/ore_sample.jpg`

- [ ] **Step 1: Create examples directory and placeholder images**

Run this Python script from the project root:

```bash
uv run python - <<'EOF'
import os
import numpy as np
from PIL import Image

os.makedirs("huggingface_space/examples", exist_ok=True)

# Defect sample: gray surface with random noise
rng = np.random.default_rng(42)
defect_img = rng.integers(50, 100, (200, 200, 3), dtype=np.uint8)
Image.fromarray(defect_img).save("huggingface_space/examples/defect_sample.jpg")

# Ore sample: brownish tones (reduce blue channel)
ore_img = rng.integers(80, 150, (200, 200, 3), dtype=np.uint8)
ore_img[:, :, 2] = ore_img[:, :, 2] // 3
Image.fromarray(ore_img).save("huggingface_space/examples/ore_sample.jpg")

print("Examples created: huggingface_space/examples/")
EOF
```

Expected output: `Examples created: huggingface_space/examples/`

- [ ] **Step 2: Verify files exist**

```bash
ls huggingface_space/examples/
```

Expected: `defect_sample.jpg  ore_sample.jpg`

- [ ] **Step 3: Commit**

```bash
git add huggingface_space/examples/
git commit -m "feat: add placeholder example images for Gradio HF Space"
```

---

### Task 5: Rewrite huggingface_space/app.py (Gradio HF Space)

**Files:**
- Modify: `huggingface_space/app.py` (full rewrite)

Key facts:
- Current file uses broken `gr.TabbedInterface` nesting inside `gr.Blocks` — entirely replaced
- `gr.Blocks` + `gr.Tabs` + `gr.Tab` is the correct pattern
- `demo.queue()` required before `demo.launch()` for API access
- API routes: `gr.Blocks` exposes `/run/predict` per event handler (not a single `/api/predict`)
- `run_defect_detection(image: PIL.Image, conf: float) -> (PIL.Image, str)` — returns annotated image + markdown
- `run_ore_classification(image: PIL.Image, conf: float) -> (PIL.Image, str)` — returns annotated image + markdown
- Model path: `yolo11s-seg.pt` (try relative to script, then project root)
- Demo mode: return original image + warning message if model not available
- `gr.Examples` paths are relative to the Gradio app file location

- [ ] **Step 1: Rewrite huggingface_space/app.py**

Replace entire contents with:

```python
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
            gr.Markdown(
                "Yüzey kusurlarını tespit eder, anomali skoru ve bakım önerisi üretir."
            )
            with gr.Row():
                with gr.Column():
                    defect_img_input = gr.Image(
                        type="pil", label="Görüntü Yükle"
                    )
                    defect_conf = gr.Slider(
                        0.1, 1.0, value=0.25, step=0.05, label="Güven Eşiği"
                    )
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
                    ore_img_input = gr.Image(
                        type="pil", label="Görüntü Yükle"
                    )
                    ore_conf = gr.Slider(
                        0.1, 1.0, value=0.25, step=0.05, label="Güven Eşiği"
                    )
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
```

- [ ] **Step 2: Verify Gradio app starts locally**

```bash
cd huggingface_space && uv run python app.py
```

Expected: Gradio server starts, prints `Running on local URL: http://127.0.0.1:7860`. Stop with Ctrl+C after confirming.

- [ ] **Step 3: Run existing tests (from project root)**

```bash
cd .. && uv run pytest tests/ -q
```

Expected: 29 passed, 0 failed

- [ ] **Step 4: Commit**

```bash
git add huggingface_space/app.py
git commit -m "feat: rebuild Gradio HF Space with gr.Blocks, two tabs, API accordion"
```

---

## Final Verification

- [ ] **Run full test suite from project root**

```bash
uv run pytest tests/ -q
```

Expected: 29 passed, 0 failed

- [ ] **Apply code formatting**

```bash
uv run black app.py pages/defect.py pages/ore.py huggingface_space/app.py
uv run isort app.py pages/defect.py pages/ore.py huggingface_space/app.py
```

Then commit if any changes:

```bash
git add app.py pages/defect.py pages/ore.py huggingface_space/app.py
git commit -m "style: apply Black and isort formatting to rebuilt UI files"
```

- [ ] **Push to main**

```bash
git push origin main
```
