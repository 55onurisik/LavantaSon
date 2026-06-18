# -*- coding: utf-8 -*-
"""
Panel — genel bakış kontrol paneli.

Analiz geçmişinden (SQLite) toplu KPI'lar ve son analizlerin özeti.
Ömer'in Dashboard.jsx sayfasının Streamlit karşılığıdır (gerçek verilerle).
"""

import pandas as pd
import streamlit as st

from lavanta import theme, history, refdata

theme.inject()

st.title("Kontrol Paneli")
st.caption("Sistem genel bakışı ve birikmiş analiz istatistikleri")

records = history.get_all_analyses()

# ── Üst KPI şeridi ────────────────────────────────────────────────────────────
total = len(records)
if total == 0:
    theme.tip("Henüz analiz yok. <b>Analiz</b> sayfasından bir görüntü yükleyerek "
              "başlayın; sonuçlar burada birikecek.", kind="info")
else:
    df = pd.DataFrame(records)
    k = st.columns(4)
    k[0].metric("Toplam Analiz", total)
    k[1].metric("Toplam Olgun Bitki", int(df["mature_count"].sum()))
    k[2].metric("Toplam Uçucu Yağ", f"{df['oil_estimate'].sum():.2f} kg")
    k[3].metric("Toplam Tahmini Gelir", f"{df['revenue_euro'].sum():.0f} €")

    k = st.columns(4)
    k[0].metric("Ort. Hasat Hazırlık", f"%{df['harvest_readiness'].mean():.1f}")
    k[1].metric("Ort. Yoğunluk", f"%{(df['density_score'].mean()*100):.1f}")
    k[2].metric("Ort. İşlem Süresi", f"{df['processing_time'].mean():.2f} s")
    k[3].metric("Ort. Rekolte", f"{df['estimated_yield'].mean():.1f} kg")

# ── Model durumu ──────────────────────────────────────────────────────────────
theme.section("Model Durumu")
summ = refdata.model_summary()
mc = st.columns(4)
if summ:
    mc[0].metric("YOLOv11 mAP@50", f"{summ['map50']:.3f}")
    mc[1].metric("Precision", f"{summ['precision']:.3f}")
    mc[2].metric("Recall", f"{summ['recall']:.3f}")
    mc[3].metric("Eğitim Epoch", summ["epochs"])
else:
    st.info("Eğitim metrikleri (results.csv) bulunamadı.")

# ── Son analizler ─────────────────────────────────────────────────────────────
if total > 0:
    theme.section("Son Analizler")
    recent = df.head(8)[[
        "date", "filename", "mature_count", "immature_count",
        "estimated_yield", "oil_estimate", "revenue_euro", "harvest_readiness",
    ]].rename(columns={
        "date": "Tarih", "filename": "Dosya", "mature_count": "Olgun",
        "immature_count": "Yetişmemiş", "estimated_yield": "Rekolte (kg)",
        "oil_estimate": "Yağ (kg)", "revenue_euro": "Gelir (€)",
        "harvest_readiness": "Hasat %",
    })
    st.dataframe(recent, width="stretch", hide_index=True)

    theme.section("Rekolte Trendi")
    trend = df.sort_values("date")[["date", "estimated_yield", "oil_estimate"]].rename(
        columns={"estimated_yield": "Rekolte (kg)", "oil_estimate": "Yağ (kg)"}
    ).set_index("date")
    st.line_chart(trend, height=260)
