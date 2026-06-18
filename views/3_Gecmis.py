# -*- coding: utf-8 -*-
"""
Geçmiş — kaydedilmiş tüm analizler.

SQLite'tan okunan analiz kayıtlarını tablo + grafik olarak gösterir,
CSV dışa aktarma ve geçmişi temizleme imkânı sunar.
"""

import pandas as pd
import streamlit as st

from lavanta import theme, history

theme.inject()

st.title("Analiz Geçmişi")
st.caption("Kaydedilmiş tüm analizler (en yeni en üstte)")

records = history.get_all_analyses()

if not records:
    theme.tip("Henüz kayıt yok. <b>Analiz</b> sayfasından bir görüntü analiz edin.", kind="info")
    st.stop()

df = pd.DataFrame(records)

# ── Özet ─────────────────────────────────────────────────────────────────────
k = st.columns(4)
k[0].metric("Kayıt Sayısı", len(df))
k[1].metric("Toplam Rekolte", f"{df['estimated_yield'].sum():.1f} kg")
k[2].metric("Toplam Yağ", f"{df['oil_estimate'].sum():.2f} kg")
k[3].metric("Toplam Gelir", f"{df['revenue_euro'].sum():.0f} €")

# ── Tablo ────────────────────────────────────────────────────────────────────
theme.section("Tüm Kayıtlar")
view = df[[
    "id", "date", "filename", "mature_count", "immature_count", "wild_count",
    "estimated_yield", "oil_estimate", "revenue_euro", "density_score",
    "harvest_readiness", "processing_time",
]].rename(columns={
    "id": "ID", "date": "Tarih", "filename": "Dosya", "mature_count": "Olgun",
    "immature_count": "Yetişmemiş", "wild_count": "Yabani", "estimated_yield": "Rekolte (kg)",
    "oil_estimate": "Yağ (kg)", "revenue_euro": "Gelir (€)", "density_score": "Yoğunluk",
    "harvest_readiness": "Hasat %", "processing_time": "Süre (s)",
})
st.dataframe(view, width="stretch", hide_index=True)

# ── Grafikler ────────────────────────────────────────────────────────────────
theme.section("Dağılımlar")
g1, g2 = st.columns(2)
with g1:
    st.caption("Analiz başına rekolte (kg)")
    st.bar_chart(df.set_index("id")["estimated_yield"], height=240)
with g2:
    st.caption("Hasat hazırlık oranı (%)")
    st.bar_chart(df.set_index("id")["harvest_readiness"], height=240)

# ── İşlemler ─────────────────────────────────────────────────────────────────
theme.section("İşlemler")
c1, c2 = st.columns(2)
c1.download_button(
    "CSV olarak indir",
    data=view.to_csv(index=False).encode("utf-8-sig"),
    file_name="lavanta_analiz_gecmisi.csv",
    mime="text/csv",
    width="stretch",
)
with c2:
    if st.button("Geçmişi Temizle", type="secondary", width="stretch"):
        if st.session_state.get("confirm_clear"):
            history.clear_history()
            st.session_state["confirm_clear"] = False
            st.success("Geçmiş temizlendi.")
            st.rerun()
        else:
            st.session_state["confirm_clear"] = True
            st.warning("Emin misiniz? Silmek için tekrar tıklayın.")
