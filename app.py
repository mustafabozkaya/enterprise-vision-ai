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
    st.markdown("""
        1. Sol menüden **Defekt Tespiti** sayfasına gidin
        2. JPG / JPEG / PNG formatında görüntü yükleyin
        3. Güven eşiğini sidebar'dan ayarlayın (varsayılan: %25)
        4. **Analiz Et** butonuna tıklayın
        5. Annotated görüntü, metrikler ve tespit tablosu görünecektir
        """)

with st.expander("📖 Cevher Ön Seçimi — Nasıl Kullanılır?"):
    st.markdown("""
        1. Sol menüden **Cevher Ön Seçimi** sayfasına gidin
        2. Cevher görüntüsü yükleyin (JPG, JPEG, PNG)
        3. Güven eşiğini sidebar'dan ayarlayın
        4. **Analiz Et** butonuna tıklayın
        5. Metal oranı, diverter önerisi ve sınıf dağılım grafiği görünecektir
        """)

# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align:center; color:#666; font-size:12px;'>© 2026 Enterprise Vision AI</div>",
    unsafe_allow_html=True,
)
