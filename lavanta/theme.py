"""
Ortak görsel tema ve küçük UI yardımcıları.

Beğenilen, sade ve açık (light) Streamlit görünümünü korur; üzerine hafif bir
lavanta vurgusu ekler. Ömer'in koyu "uydu telemetri" teması bilinçli olarak
kullanılmaz (beğenilmeyen taraf oydu).
"""

import streamlit as st

LAVENDER = "#7c3aed"
LAVENDER_SOFT = "#f5f3ff"

_CSS = """
<style>
:root { --lav: #7c3aed; --lav-soft: #f5f3ff; }

/* Genel tipografi ve genişlik */
.block-container { padding-top: 2.2rem; max-width: 1200px; }

/* Başlıklar */
h1, h2, h3 { letter-spacing: -0.01em; }

/* Bölüm başlığı şeridi */
.lav-section {
    border-left: 4px solid var(--lav);
    background: var(--lav-soft);
    padding: 0.55rem 0.9rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0 0.9rem 0;
}
.lav-section .t { font-weight: 700; font-size: 1.02rem; color: #2e1065; }
.lav-section .s { font-size: 0.8rem; color: #6d28d9; opacity: 0.85; }

/* st.metric kartları */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #ece9fb;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 2px rgba(124,58,237,0.05);
}
[data-testid="stMetricValue"] { color: #2e1065; }

/* Sidebar başlık tonu */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #4c1d95; }

/* Rozet */
.lav-badge {
    display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:0.72rem; font-weight:600; letter-spacing:.02em;
}

/* Öneri kutuları */
.lav-tip { border-radius:10px; padding:0.7rem 0.9rem; font-size:0.9rem; margin-bottom:0.5rem; }
.lav-tip.ok   { background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; }
.lav-tip.warn { background:#fffbeb; border:1px solid #fde68a; color:#92400e; }
.lav-tip.info { background:#eff6ff; border:1px solid #bfdbfe; color:#1e40af; }
.lav-tip b { font-weight:700; }

/* Buton vurgusu */
.stDownloadButton button, .stButton button { border-radius: 10px; }
</style>
"""


def inject():
    """Tema CSS'ini sayfaya enjekte eder. Her sayfanın başında çağrılır."""
    st.markdown(_CSS, unsafe_allow_html=True)


def section(title: str, subtitle: str = ""):
    """Lavanta vurgulu bölüm başlığı."""
    sub = f"<div class='s'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='lav-section'><div class='t'>{title}</div>{sub}</div>",
        unsafe_allow_html=True,
    )


def tip(text: str, kind: str = "info"):
    """Renkli öneri/uyarı kutusu. kind: ok | warn | info."""
    st.markdown(f"<div class='lav-tip {kind}'>{text}</div>", unsafe_allow_html=True)


def badge(text: str, color: str = LAVENDER, bg: str = LAVENDER_SOFT) -> str:
    """Satır içi rozet HTML'i döner."""
    return f"<span class='lav-badge' style='color:{color};background:{bg}'>{text}</span>"
