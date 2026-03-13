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
    analyze_clicked = btn_col.button("🔍 Analiz Et", type="primary", use_container_width=True)

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

        st.info(res["recommendation"])

        if res["rows"]:
            st.dataframe(create_metrics_dataframe(res["rows"]), use_container_width=True)
        else:
            st.info("Bu görüntüde tespit bulunamadı.")
