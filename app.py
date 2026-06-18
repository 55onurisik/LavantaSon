# -*- coding: utf-8 -*-
"""
Lavanta AI — birleşik uygulama (router).

Beğenilen Streamlit ön yüzü + `omer/` Rekolte Tahmini projesinin gerçek
özellikleri tek bir çok sayfalı Streamlit uygulamasında birleştirilmiştir.

Çalıştırma:
    streamlit run app.py
"""

import streamlit as st

from lavanta import theme, history

st.set_page_config(
    page_title="Lavanta AI — Olgunluk & Rekolte",
    page_icon="🌿",
    layout="wide",
)

theme.inject()
history.init_db()  # geçmiş tablosunu hazırla

# ── Sayfalar ─────────────────────────────────────────────────────────────────
analiz  = st.Page("views/1_Analiz.py",       title="Analiz",             icon="🔬", default=True)
panel   = st.Page("views/2_Panel.py",        title="Panel",              icon="📊")
gecmis  = st.Page("views/3_Gecmis.py",       title="Geçmiş",             icon="🗂️")
modelp  = st.Page("views/4_Model.py",        title="Model Performansı",  icon="🎯")
veri    = st.Page("views/5_VeriSeti.py",     title="Veri Seti",          icon="🗃️")
ozet    = st.Page("views/6_ProjeOzeti.py",   title="Proje Özeti",        icon="📋")
yabani  = st.Page("views/7_YabaniOtTespit.py", title="Yabani Ot Tespiti", icon="🌿")

pg = st.navigation({
    "Analiz":  [analiz, panel, gecmis],
    "Sistem":  [modelp, veri],
    "Bitirme": [ozet, yabani],
})

st.sidebar.divider()
st.sidebar.caption("🌿 Lavanta AI • YOLOv11 + Rekolte Tahmini")

pg.run()
