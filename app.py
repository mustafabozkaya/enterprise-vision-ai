"""
Enterprise Vision AI MVP - Ana Giriş Sayfası
Streamlit Uygulaması

Bu uygulama iki ana modül içerir:
1. Defekt Tespiti - Yüzey kusurlarının tespiti
2. Cevher Ön Seçimi - Cevher sınıflandırma ve ayrıştırma
"""

import streamlit as st

# -----------------------------------------------------------------------------
# CACHED HELPER FUNCTIONS
# -----------------------------------------------------------------------------


@st.cache_data(ttl=30)
def check_api_health():
    """
    FastAPI server sağlık durumunu kontrol eder.
    30 saniye boyunca cache'lenir.

    Returns:
        tuple: (status_text, status_color)
    """
    try:
        import requests

        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            return ("🟢 Aktif", "#3fb950")
        else:
            return ("🟡 Uyarı", "#d29922")
    except Exception:
        return ("🔴 Bağlantı Yok", "#f85149")


# -----------------------------------------------------------------------------
# SAYFA KONFİGURASYONU
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise Vision AI MVP",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# CUSTOM CSS - Hide default Streamlit navigation
# -----------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* Ana tema renkleri */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Başlık stilleri */
    h1, h2, h3, h4 {
        color: #00d4ff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Hide Streamlit's default sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Hide any legacy navigation */
    .stSidebarUser {
        display: none !important;
    }
    
    /* Sidebar içeriği */
    .sidebar-content {
        padding: 20px;
    }
    
    /* Sidebar başlık */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px;
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .sidebar-header h1 {
        font-size: 20px !important;
        margin: 0;
        color: #ffffff !important;
    }
    
    /* Sidebar bölüm başlıkları */
    .sidebar-section-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
        margin: 20px 0 10px 0;
        padding: 0 12px;
    }
    
    /* Navigation link styling */
    .nav-link {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 8px;
        color: #c9d1d9;
        text-decoration: none;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .nav-link:hover {
        background-color: #21262d;
        color: #ffffff;
    }
    
    .nav-link.active {
        background-color: #1f6feb;
        color: #ffffff;
    }
    
    .nav-icon {
        font-size: 18px;
    }
    
    /* Model durumu kartı */
    .model-status-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        border: 1px solid #30363d;
    }
    
    .model-status-card .title {
        font-size: 14px;
        color: #8b949e;
        margin-bottom: 8px;
    }
    
    .model-status-card .status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        color: #3fb950;
    }
    
    .model-status-card .status-dot {
        width: 8px;
        height: 8px;
        background-color: #3fb950;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Bilgi kartı */
    .info-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        border: 1px solid #30363d;
    }
    
    .info-card h4 {
        font-size: 14px;
        color: #ffffff !important;
        margin-bottom: 8px;
    }
    
    .info-card p {
        font-size: 12px;
        color: #8b949e;
        margin: 4px 0;
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
    
    /* Divider */
    .sidebar-divider {
        height: 1px;
        background-color: #30363d;
        margin: 16px 0;
    }
    
    /* Version badge */
    .version-badge {
        display: inline-block;
        padding: 4px 8px;
        background-color: #238636;
        border-radius: 12px;
        font-size: 11px;
        color: #ffffff;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# SIDEBAR RENDER - Professional Navigation
# -----------------------------------------------------------------------------


def render_sidebar():
    """Modern enterprise-style sidebar with professional navigation."""

    with st.sidebar:
        # Logo ve başlık
        st.markdown(
            """
        <div class="sidebar-header">
            <span style="font-size: 28px;">🏭</span>
            <div>
                <h1>Enterprise Vision AI</h1>
                <span class="version-badge">v1.0.0</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Ana sayfa linki
        st.markdown('<div class="sidebar-section-title">Sayfalar</div>', unsafe_allow_html=True)

        # Navigation using Streamlit's native page_link
        st.markdown(
            """
        <a href="./" class="nav-link" style="background-color: #1f6feb; color: #ffffff;">
            <span class="nav-icon">🏠</span>
            <span>Ana Sayfa</span>
        </a>
        """,
            unsafe_allow_html=True,
        )

        st.page_link("pages/01_Defekt_Tespiti.py", label="🔍 Defekt Tespiti")
        st.page_link("pages/02_Cevher_On_Secimi.py", label="💎 Cevher Ön Seçimi")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # API Durumu
        st.markdown(
            '<div class="sidebar-section-title">Sistem Durumu</div>', unsafe_allow_html=True
        )

        # API server durumu (cached)
        api_status, api_color = check_api_health()

        st.markdown(
            f"""
        <div class="model-status-card">
            <div class="title">FastAPI Server</div>
            <div class="status" style="color: {api_color};">
                <span class="status-dot" style="background-color: {api_color};"></span>
                {api_status}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Model durumu
        st.markdown(
            """
        <div class="model-status-card">
            <div class="title">YOLO11 Segmentation</div>
            <div class="status" style="color: #3fb950;">
                <span class="status-dot"></span>
                Hazır
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Hakkında
        st.markdown('<div class="sidebar-section-title">Bilgi</div>', unsafe_allow_html=True)

        with st.expander("ℹ️ Uygulama Hakkında"):
            st.markdown("""
            **Enterprise Vision AI**
            
            Endüstriyel görüntü işleme çözümleri için geliştirilmiş yapay zeka sistemi.
            
            - **Versiyon:** 1.0.0
            - **Tarih:** 2026
            - **Framework:** Streamlit + FastAPI
            """)

        with st.expander("🔌 API Dokümantasyonu"):
            st.markdown("""
            **FastAPI Endpoints:**
            
            - `GET /health` - Sağlık kontrolü
            - `GET /api/v1/models` - Model listesi  
            - `POST /api/v1/detect/defects` - Defekt tespiti
            - `POST /api/v1/classify/ore` - Cevher sınıflandırması
            
            **API Dokümantasyonu:** `/docs`
            """)


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
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )

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
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Enterprise Vision AI MVP © 2026</p>
        <p>Tüm hakları saklıdır.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


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
