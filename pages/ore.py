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
    st.markdown("Maden cevherlerini sınıflandırır, metal oranı hesaplar ve diverter yönü önerir.")

    # --- Yükleme ---
    uploaded_file = st.file_uploader(
        "Görüntü Yükle", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    # --- Analiz Et butonu ---
    _, btn_col, _ = st.columns([1, 2, 1])
    analyze_clicked = btn_col.button("💎 Analiz Et", type="primary", use_container_width=True)

    # --- Boş durum ---
    if uploaded_file is None:
        st.info("Analiz için bir cevher görüntüsü yükleyin (JPG, JPEG, PNG).")
        st.markdown("**Metal Oranı Formülü:** `(manyetit + krom) / toplam_tespit × 100`")
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

        counts = (
            calculate_ore_metrics(results)
            if results
            else {"manyetit": 0, "krom": 0, "atık": 0, "düşük tenör": 0}
        )
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

        st.info(res["diverter"])

        df = create_ore_dataframe(res["counts"])
        st.bar_chart(df.set_index("Class")["Count"])
