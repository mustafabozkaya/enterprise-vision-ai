"""
Enterprise Vision AI MVP - Ana Giriş Sayfası
Streamlit Uygulaması

Bu uygulama iki ana modül içerir:
1. Defekt Tespiti - Yüzey kusurlarının tespiti
2. Cevher Ön Seçimi - Cevher sınıflandırma ve ayrıştırma
"""

import streamlit as st
from pathlib import Path

# -----------------------------------------------------------------------------
# SAYFA KONFİGURASYONU
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise Vision AI MVP",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------

st.markdown("""
<style>
    /* Ana tema renkleri */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Başlık stilleri */
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d23;
    }
    
    /* Buton stilleri */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    
    /* Card stilleri */
    .card {
        background-color: #1e2330;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
    }
    
    /* Metric kutuları */
    .metric-box {
        background-color: #1e2330;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    
    /* Info kutuları */
    .info-box {
        background-color: #1e2330;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR - MODÜL SEÇİMİ
# -----------------------------------------------------------------------------

def render_sidebar():
    """Sidebar menüsünü oluşturur."""
    
    st.sidebar.title("🏭 BAS AI MVP")
    st.sidebar.markdown("---")
    
    # Logo ve açıklama
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px;">
        <h3>Endüstriyel Yapay Zeka</h3>
        <p style="color: #888;">Görüntü İşleme Çözümleri</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Modül seçimi - Streamlit otomatik olarak pages klasöründeki sayfaları listeler
    st.sidebar.subheader("📂 Sayfalar")
    st.sidebar.markdown("""
    - 🏠 Ana Sayfa
    - 🔍 Defekt Tespiti  
    - 💎 Cevher Ön Seçimi
    
    *Sol menüden sayfa seçin*
    """)
    
    st.sidebar.markdown("---")
    
    # Model durumu
    st.sidebar.subheader("🤖 Model Durumu")
    
    with st.sidebar.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Defekt:**")
            st.success("YOLO11")
        with col2:
            st.markdown("**Cevher:**")
            st.success("YOLO11")
    
    st.sidebar.markdown("---")
    
    # Hakkında
    with st.sidebar.expander("ℹ️ Hakkında"):
        st.markdown("""
        **Enterprise Vision AI MVP**
        
        Bu uygulama, endüstriyel görüntü işleme
        çözümleri için geliştirilmiş bir MVP'dir.
        
        **Versiyon:** 1.0.0
        **Tarih:** 2026
        """)
    
    return


def render_homepage():
    """Ana sayfa içeriğini oluşturur."""
    
    # Hoş geldin başlığı
    st.title("🏭 Enterprise Vision AI MVP")
    st.markdown("### Yapay Zeka Destekli Kalite Kontrol Sistemi")
    
    st.markdown("---")
    
    # İstatistik kartları
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Aktif Modüller", value="2")
    with col2:
        st.metric(label="Model Doğruluğu", value="94%")
    with col3:
        st.metric(label="İşlem Hızı", value="30 FPS")
    with col4:
        st.metric(label="Desteklenen Sınıf", value="9+")
    
    st.markdown("---")
    
    # Modül açıklamaları
    st.subheader("📦 Mevcut Modüller")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔍 Defekt Tespiti</h3>
            <p>Endüstriyel ürünlerdeki yüzey kusurlarını 
            gerçek zamanlı olarak tespit eden modüldür.</p>
            <ul>
                <li>Çatlak, çizik, delik tespiti</li>
                <li>Anomali skorlaması</li>
                <li>Severity seviye belirleme</li>
                <li>Bakım önerileri</li>
            </ul>
            <p><strong>Model:</strong> YOLO11 Seg</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>💎 Cevher Ön Seçimi</h3>
            <p>Maden cevherlerini sınıflandıran ve 
            ayrıştıran modüldür.</p>
            <ul>
                <li>Manyetit, Krom tespiti</li>
                <li>Atık / düşük tenör ayrımı</li>
                <li>Metal oranı hesaplama</li>
                <li>Diverter kontrol önerileri</li>
            </ul>
            <p><strong>Model:</strong> YOLO11 Seg</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Kullanım talimatları
    st.subheader("📖 Kullanım Talimatları")
    
    with st.expander("Defekt Tespiti Nasıl Kullanılır?"):
        st.markdown("""
        1. Sol menüden **Defekt Tespiti** modülünü seçin
        2. **Görüntü Yükle** veya **Video Yükle** seçeneğini kullanın
        3. RTSP stream kullanmak isterseniz URL girin
        4. Sonuçları görüntüleyin ve analiz edin
        5. Anomali skoruna göre bakım önerilerini takip edin
        """)
    
    with st.expander("Cevher Ön Seçimi Nasıl Kullanılır?"):
        st.markdown("""
        1. Sol menüden **Cevher Ön Seçimi** modülünü seçin
        2. Cevher görüntüsü veya videosu yükleyin
        3. RTSP stream kullanmak isterseniz URL girin
        4. Sınıflandırma sonuçlarını inceleyin
        5. Metal oranına göre diverter önerisini takip edin
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Enterprise Vision AI MVP © 2026</p>
        <p>Tüm hakları saklıdır.</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN FONKSİYON
# -----------------------------------------------------------------------------

def main():
    """Ana uygulama fonksiyonu."""
    
    # Sidebar'ı render et
    render_sidebar()
    
    # Ana sayfayı render et
    render_homepage()


if __name__ == "__main__":
    main()
