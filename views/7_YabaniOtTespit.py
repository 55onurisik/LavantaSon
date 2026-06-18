# -*- coding: utf-8 -*-
"""
Yabani Ot Tespiti — BitirmeProjesi modeli (3 sınıf).

YOLOv11n — olgunlasmis / yabani_bitki / yetismemis
Model: runs/mustafa_model/best.pt  (5.5 MB, mAP@0.50: %55.8)
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image
import tempfile
import os

from lavanta import theme, detection as det

theme.inject()

MUSTAFA_MODEL = Path(__file__).resolve().parent.parent / "runs" / "mustafa_model" / "best.pt"

SINIF_RENK = {
    "olgunlasmis":  {"hex": "#7c3aed", "bg": "#f5f3ff", "label": "Olgunlaşmış"},
    "yabani_bitki": {"hex": "#dc2626", "bg": "#fef2f2", "label": "Yabani Ot"},
    "yetismemis":   {"hex": "#d97706", "bg": "#fffbeb", "label": "Yetişmemiş"},
}

st.title("Yabani Ot Tespiti")
st.caption("YOLOv11n (BitirmeProjesi) · 3 sınıf · mAP@0.50: %55.8")

# ── Model kontrolü ────────────────────────────────────────────────────────────
if not MUSTAFA_MODEL.exists():
    st.error(f"Model bulunamadı: `{MUSTAFA_MODEL}`")
    st.stop()

model = det.load_model(str(MUSTAFA_MODEL))
sinif_isimleri = det.model_class_names(model)  # {0: 'olgunlasmis', 1: ...}

# ── Sidebar: ayarlar ─────────────────────────────────────────────────────────
st.sidebar.markdown("## Tespit Ayarları")
conf = st.sidebar.slider("Güven eşiği", 0.10, 0.90, 0.25, 0.05,
                          help="Düşürürsen daha fazla tespit yapar, yanlış alarm artar.")
iou  = st.sidebar.slider("IoU eşiği (NMS)", 0.30, 0.90, 0.45, 0.05,
                          help="Üst üste gelen kutucukların elenmesi.")
st.sidebar.divider()

# ── Görüntü yükle ────────────────────────────────────────────────────────────
theme.section("Görüntü Yükle", "Drone görüntüsü yükle — model yabancı otları tespit etsin")
yuklu = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if yuklu is None:
    theme.tip("Analiz için yukarıdan bir drone görüntüsü yükleyin.", kind="info")
    st.stop()

pil_img = Image.open(yuklu).convert("RGB")
img_np  = np.array(pil_img)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# ── Inference ────────────────────────────────────────────────────────────────
file_id = yuklu.file_id
if st.session_state.get("mustafa_file_id") != file_id:
    with st.spinner("Yabani ot tespiti yapılıyor..."):
        results = model.predict(
            source=img_bgr, conf=conf, iou=iou, imgsz=640, verbose=False
        )
        st.session_state["mustafa_result"]  = results[0]
        st.session_state["mustafa_img_bgr"] = img_bgr
        st.session_state["mustafa_file_id"] = file_id

result  = st.session_state["mustafa_result"]
img_bgr = st.session_state["mustafa_img_bgr"]

# ── Sonuç görseli ────────────────────────────────────────────────────────────
annotated_rgb = result.plot()[:, :, ::-1]

theme.section("Tespit Sonucu", f"{yuklu.name} · {len(result.boxes) if result.boxes else 0} nesne")

col_o, col_p = st.columns(2)
with col_o:
    st.subheader("Orijinal")
    st.image(pil_img, use_container_width=True)
with col_p:
    st.subheader("Tespit")
    st.image(annotated_rgb, use_container_width=True)

if result.boxes is not None and len(result.boxes) > 0:
    st.download_button(
        "Tespit sonucunu indir (JPG)",
        data=cv2.imencode(".jpg", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name=f"yabani_tespit_{yuklu.name}",
        mime="image/jpeg",
    )

# ── Tespit özeti ─────────────────────────────────────────────────────────────
theme.section("Tespit Özeti")

if result.boxes is None or len(result.boxes) == 0:
    theme.tip("Bu görselde tespit yapılamadı. Güven eşiğini düşürmeyi deneyin.", kind="warn")
    st.stop()

boxes = result.boxes
sayim = {}
tespitler = []
for i in range(len(boxes)):
    cls_id  = int(boxes.cls[i].item())
    conf_v  = float(boxes.conf[i].item())
    ad      = sinif_isimleri.get(cls_id, str(cls_id))
    sayim[ad] = sayim.get(ad, 0) + 1
    tespitler.append({"#": i + 1, "Sınıf": ad, "Güven (%)": f"{conf_v * 100:.1f}"})

# Metrik kartları
cols = st.columns(len(SINIF_RENK) + 1)
cols[0].metric("Toplam Tespit", len(boxes))
for i, (sinif, stil) in enumerate(SINIF_RENK.items()):
    adet = sayim.get(sinif, 0)
    border = f"border: 2px solid {stil['hex']};" if sinif == "yabani_bitki" else ""
    cols[i + 1].markdown(
        f"<div style='background:{stil['bg']};{border}border-radius:12px;"
        f"padding:14px;text-align:center;'>"
        f"<div style='font-size:1.8rem;font-weight:700;color:{stil['hex']};'>{adet}</div>"
        f"<div style='font-size:0.8rem;color:#444;margin-top:2px;'>{stil['label']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Sınıf dağılımı
if sayim:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Sınıf Dağılımı**")
        for sinif, adet in sayim.items():
            stil = SINIF_RENK.get(sinif, {"hex": "#666", "bg": "#f9f9f9", "label": sinif})
            pct  = adet / len(boxes) * 100
            st.markdown(
                f"<div style='margin-bottom:8px;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:0.83rem;margin-bottom:3px;'>"
                f"<span style='color:{stil['hex']};font-weight:600;'>{stil['label']}</span>"
                f"<span style='color:#444;'>{adet} tespit</span></div>"
                f"<div style='background:#e5e7eb;border-radius:999px;height:10px;overflow:hidden;'>"
                f"<div style='width:{pct:.0f}%;height:100%;background:{stil['hex']};"
                f"border-radius:999px;'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown("**Tespit Listesi**")
        st.dataframe(pd.DataFrame(tespitler), hide_index=True, use_container_width=True)

# ── Model Performansı ────────────────────────────────────────────────────────
st.divider()
theme.section("Model Performansı (YMH455 Bitirme Projesi)", "2 eğitim — veri temizliği +%19.1 mAP artışı")

metrikler = [
    ("mAP@0.50",  36.7, 55.8),
    ("Precision", 40.9, 55.2),
    ("Recall",    42.4, 57.1),
    ("F1 Score",  41.6, 56.1),
]

col_m, col_g = st.columns([2, 3])

with col_m:
    for isim, ham, temiz in metrikler:
        delta = temiz - ham
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #ece9fb;"
            f"border-radius:12px;padding:12px 15px;margin-bottom:8px;"
            f"box-shadow:0 1px 2px rgba(124,58,237,0.05);'>"
            f"<div style='display:flex;justify-content:space-between;'>"
            f"<b style='color:#2e1065;font-size:0.9rem;'>{isim}</b>"
            f"<span style='color:#16a34a;font-weight:700;'>+{delta:.1f}%</span></div>"
            f"<div style='font-size:0.72rem;color:#888;margin:5px 0 2px;'>1. Eğitim (Ham)</div>"
            f"<div style='background:#e5e7eb;border-radius:999px;height:8px;overflow:hidden;'>"
            f"<div style='width:{ham}%;height:100%;background:#9ca3af;border-radius:999px;'></div></div>"
            f"<div style='font-size:0.72rem;color:#888;margin:5px 0 2px;'>2. Eğitim (Temiz)</div>"
            f"<div style='background:#ede9fe;border-radius:999px;height:8px;overflow:hidden;'>"
            f"<div style='width:{temiz}%;height:100%;background:#7c3aed;border-radius:999px;'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

with col_g:
    grafik = pd.DataFrame(
        {"1. Eğitim (Ham)": [m[1] for m in metrikler],
         "2. Eğitim (Temiz)": [m[2] for m in metrikler]},
        index=[m[0] for m in metrikler],
    )
    st.bar_chart(grafik, height=240, color=["#9ca3af", "#7c3aed"])

    theme.tip(
        "<b>Temel Bulgu:</b> 39.119 gereksiz etiket ve 8.434 hatalı bbox temizlenerek "
        "mAP <b>%36.7 → %55.8</b> (+%19.1) yükseldi.",
        kind="ok",
    )

    st.markdown("**Sınıf Bazlı mAP@0.50**")
    sinif_map = pd.DataFrame(
        {"mAP (%)": [81.2, 41.4, 44.9]},
        index=["olgunlasmis", "yabani_bitki", "yetismemis"],
    )
    st.bar_chart(sinif_map, height=180, color="#7c3aed")
