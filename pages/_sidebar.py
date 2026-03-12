"""
Shared sidebar component for all Streamlit pages.
Renders the common navigation, branding, and system status.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_API_BASE = os.environ.get("API_URL", "").rstrip("/")


def _detect_api_url() -> str:
    import requests

    for port in (8000, 8001):
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=1)
            if r.status_code == 200:
                return f"http://localhost:{port}"
        except Exception:
            pass
    return "http://localhost:8000"


def get_api_url() -> str:
    return _API_BASE if _API_BASE else _detect_api_url()


@st.cache_data(ttl=30)
def check_api_health():
    try:
        import requests

        url = get_api_url()
        response = requests.get(f"{url}/health", timeout=2)
        if response.status_code == 200:
            return ("🟢 Aktif", "#3fb950", url)
        else:
            return ("🟡 Uyarı", "#d29922", url)
    except Exception:
        return ("🔴 Bağlantı Yok", "#f85149", "")


SIDEBAR_CSS = """
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3, h4 { color: #00d4ff !important; }
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebarNav"] { display: none !important; }
    .stSidebarUser { display: none !important; }
    .sidebar-header {
        display: flex; align-items: center; gap: 12px;
        padding: 16px;
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border-radius: 12px; margin-bottom: 20px;
    }
    .sidebar-header h1 { font-size: 20px !important; margin: 0; color: #ffffff !important; }
    .sidebar-section-title {
        font-size: 12px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px; color: #8b949e; margin: 20px 0 10px 0; padding: 0 12px;
    }
    .nav-link {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 16px; margin: 4px 0; border-radius: 8px;
        color: #c9d1d9; text-decoration: none; font-size: 14px; cursor: pointer;
        transition: all 0.2s ease;
    }
    .nav-link:hover { background-color: #21262d; color: #ffffff; }
    .nav-link.active { background-color: #1f6feb; color: #ffffff; }
    .nav-icon { font-size: 18px; }
    .model-status-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border-radius: 12px; padding: 16px; margin: 10px 0; border: 1px solid #30363d;
    }
    .model-status-card .title { font-size: 14px; color: #8b949e; margin-bottom: 8px; }
    .model-status-card .status {
        display: flex; align-items: center; gap: 8px;
        font-size: 14px; color: #3fb950;
    }
    .model-status-card .status-dot {
        width: 8px; height: 8px; background-color: #3fb950;
        border-radius: 50%; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .sidebar-divider { height: 1px; background-color: #30363d; margin: 16px 0; }
    .version-badge {
        display: inline-block; padding: 4px 8px; background-color: #238636;
        border-radius: 12px; font-size: 11px; color: #ffffff; font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        border: none; border-radius: 8px; color: white; font-weight: bold;
    }
    .card {
        background-color: #1e2330; border-radius: 12px; padding: 20px;
        margin: 10px 0; border-left: 4px solid #00d4ff;
    }
    .metric-box {
        background-color: #1e2330; border-radius: 8px;
        padding: 15px; text-align: center; margin: 5px;
    }
    .info-box { background-color: #1e2330; border-radius: 8px; padding: 15px; margin: 10px 0; }
</style>
"""


def render_sidebar(active_page: str = ""):
    """
    Renders the common sidebar: branding, navigation, and system status.

    Args:
        active_page: one of "", "defect", "ore" — highlights the active nav link
    """
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # Branding
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

        # Navigation
        st.markdown('<div class="sidebar-section-title">Sayfalar</div>', unsafe_allow_html=True)

        home_style = "background-color: #1f6feb; color: #ffffff;" if active_page == "" else ""
        st.markdown(
            f"""
        <a href="/" class="nav-link" style="{home_style}">
            <span class="nav-icon">🏠</span>
            <span>Ana Sayfa</span>
        </a>
        """,
            unsafe_allow_html=True,
        )

        st.page_link("pages/01_Defekt_Tespiti.py", label="🔍 Defekt Tespiti")
        st.page_link("pages/02_Cevher_On_Secimi.py", label="💎 Cevher Ön Seçimi")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # System status
        st.markdown(
            '<div class="sidebar-section-title">Sistem Durumu</div>', unsafe_allow_html=True
        )

        api_status, api_color, api_url = check_api_health()
        api_url_display = api_url if api_url else "—"

        st.markdown(
            f"""
        <div class="model-status-card">
            <div class="title">FastAPI Server</div>
            <div class="status" style="color: {api_color};">
                <span class="status-dot" style="background-color: {api_color};"></span>
                {api_status}
            </div>
            <div style="font-size:11px; color:#8b949e; margin-top:4px;">{api_url_display}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

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

        # Info expanders
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
            _, _, _url = check_api_health()
            docs_url = f"{_url}/docs" if _url else "http://localhost:8000/docs"
            st.markdown(f"""
            **FastAPI Endpoints:**

            - `GET /health` - Sağlık kontrolü
            - `GET /api/v1/models` - Model listesi
            - `POST /api/v1/detect/defects/file` - Defekt tespiti
            - `POST /api/v1/classify/ore/file` - Cevher sınıflandırması

            **Swagger UI:** [{docs_url}]({docs_url})
            """)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
