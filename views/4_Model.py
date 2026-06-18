# -*- coding: utf-8 -*-
"""
Model Performansı — GERÇEK eğitim çıktıları.

Ömer'in sabit (mock) grafikleri yerine, bizim eğitilmiş YOLOv11 modelimizin
gerçek `results.csv` eğrileri ve confusion matrix / PR eğrisi görselleri.
"""

import streamlit as st

from lavanta import theme, refdata, config

theme.inject()

st.title("Model Performansı")
st.caption("Eğitilmiş YOLOv11 — gerçek eğitim metrikleri (runs/lavanta_yolo11)")

summ = refdata.model_summary()
if summ is None:
    st.error(f"Eğitim sonuçları bulunamadı: `{config.RESULTS_CSV}`")
    st.stop()

# ── Final metrikler ──────────────────────────────────────────────────────────
theme.section("Final Metrikler", f"{summ['epochs']} epoch • 2 sınıf (olgunlaşmış / yetişmemiş)")
k = st.columns(4)
k[0].metric("mAP@50", f"{summ['map50']:.3f}")
k[1].metric("mAP@50-95", f"{summ['map50_95']:.3f}")
k[2].metric("Precision", f"{summ['precision']:.3f}")
k[3].metric("Recall", f"{summ['recall']:.3f}")

# ── Eğriler ──────────────────────────────────────────────────────────────────
theme.section("Metrik Eğrileri (epoch bazında)")
curves = refdata.metric_curves()
if curves is not None:
    st.line_chart(curves, height=300)

theme.section("Kayıp Eğrileri (train / val)")
losses = refdata.loss_curves()
if losses is not None:
    st.line_chart(losses, height=300)

# ── Görseller ────────────────────────────────────────────────────────────────
theme.section("Eğitim Görselleri")
g1, g2 = st.columns(2)
cm = refdata.run_image("confusion_matrix.png")
pr = refdata.run_image("BoxPR_curve.png")
if cm:
    g1.image(str(cm), caption="Confusion Matrix", width="stretch")
if pr:
    g2.image(str(pr), caption="Precision-Recall Eğrisi", width="stretch")

res = refdata.run_image("results.png")
if res:
    st.image(str(res), caption="Eğitim özeti (results.png)", width="stretch")

st.caption("Not: Bu değerler kök projedeki gerçek eğitim koşusundan alınmıştır; "
           "Ömer projesindeki örnek/sabit metrikler kullanılmamıştır.")
