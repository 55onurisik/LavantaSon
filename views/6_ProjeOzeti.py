# -*- coding: utf-8 -*-
"""
Proje Özeti — YMH455 Bitirme Projesi geliştirme hikayesi.

Veri kalitesi analizi, iki eğitim karşılaştırması ve sınıf bazlı sonuçlar.
"""

import pandas as pd
import streamlit as st

from lavanta import theme

theme.inject()

st.title("Proje Özeti")
st.caption("YMH455 Bitirme Projesi · Mustafa BAL (220541003) · 2026")

# ── Proje Hakkında ────────────────────────────────────────────────────────────
theme.section("Projenin Amacı")

c1, c2 = st.columns([3, 2])

with c1:
    st.markdown(
        "Bu proje, **DJI drone** ile çekilen lavanta tarlası görüntülerinden "
        "**yabancı ot tespiti** yapmayı amaçlamaktadır. Geliştirilen sistem; "
        "tarla sahiplerinin hangi bölgede yabancı ot bulunduğunu otomatik "
        "tespit etmesine ve tarla sağlığını uzaktan izlemesine olanak tanır."
    )
    st.markdown(
        "Çalışmada **YOLOv11n** mimarisi, **Google Colab** ortamında "
        "Tesla T4 GPU üzerinde eğitilmiştir."
    )

    theme.section("Sistem Pipeline")
    adimlar = [
        ("🚁", "DJI Drone Görüntüsü",   "Lavanta tarlası üzerinde uçuş"),
        ("🏷️", "Roboflow Etiketleme",   "Manuel bounding box çizimi"),
        ("🧹", "Veri Temizleme",         "39.119 gereksiz etiket + 8.434 hatalı bbox silindi"),
        ("🧠", "YOLOv11n Eğitimi",      "Google Colab, Tesla T4, 94 epoch"),
        ("✅", "Tespit & Analiz",        "mAP %55.8, F1 %56.1"),
    ]
    for i, (ikon, baslik, alt) in enumerate(adimlar):
        ok = "→" if i < len(adimlar) - 1 else ""
        st.markdown(
            f"<div style='background:#f5f3ff;border-left:3px solid #7c3aed;"
            f"border-radius:0 8px 8px 0;padding:9px 14px;margin-bottom:6px;'>"
            f"<span style='font-size:1.1rem;'>{ikon}</span>&nbsp;"
            f"<b style='color:#2e1065;'>{baslik}</b>&nbsp;"
            f"<span style='color:#6d28d9;font-size:0.8rem;'>— {alt}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

with c2:
    theme.section("Tespit Sınıfları")
    siniflar = [
        ("#7c3aed", "#f5f3ff", "olgunlasmis",  "Olgunlaşmış lavanta", "81.2"),
        ("#dc2626", "#fef2f2", "yabani_bitki",  "Yabancı ot / weed",   "41.4"),
        ("#d97706", "#fffbeb", "yetismemis",    "Yetişmemiş lavanta",  "44.9"),
    ]
    for clr, bg, sinif, aciklama, map_val in siniflar:
        pct = float(map_val)
        st.markdown(
            f"<div style='background:{bg};border:1px solid {clr}33;"
            f"border-radius:10px;padding:12px 14px;margin-bottom:8px;'>"
            f"<span style='color:{clr};font-weight:700;font-size:0.85rem;'>{sinif}</span>"
            f"<div style='color:#444;font-size:0.8rem;margin-top:2px;'>{aciklama}</div>"
            f"<div style='background:#e5e7eb;border-radius:999px;height:10px;"
            f"overflow:hidden;margin-top:8px;'>"
            f"<div style='width:{pct}%;height:100%;background:{clr};"
            f"border-radius:999px;'></div></div>"
            f"<div style='font-size:0.75rem;color:{clr};font-weight:600;"
            f"margin-top:3px;'>mAP@0.50: %{map_val}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    theme.section("Veri Seti (v12)")
    veri = [("Eğitim", 2285, "#7c3aed"), ("Doğrulama", 131, "#2563eb"), ("Test", 65, "#d97706")]
    toplam = 2481
    for ad, sayi, clr in veri:
        pct = sayi / toplam * 100
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.82rem;margin-bottom:2px;'>"
            f"<span style='color:#444;'>{ad}</span>"
            f"<span style='font-weight:600;color:{clr};'>{sayi:,}</span></div>"
            f"<div style='background:#e5e7eb;border-radius:999px;height:9px;"
            f"overflow:hidden;margin-bottom:8px;'>"
            f"<div style='width:{pct:.0f}%;height:100%;background:{clr};"
            f"border-radius:999px;'></div></div>",
            unsafe_allow_html=True,
        )
    st.caption(f"Toplam: {toplam:,} görsel")

# ── Eğitim Karşılaştırması ───────────────────────────────────────────────────
theme.section(
    "Eğitim Karşılaştırması",
    "Ham veri → Veri temizleme → Temiz veri: tek adım, +%19.1 mAP artışı",
)

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
            f"border-radius:12px;padding:13px 16px;margin-bottom:8px;"
            f"box-shadow:0 1px 2px rgba(124,58,237,0.05);'>"
            f"<div style='display:flex;justify-content:space-between;'>"
            f"<span style='font-weight:600;color:#2e1065;'>{isim}</span>"
            f"<span style='color:#16a34a;font-weight:700;'>+{delta:.1f}%</span>"
            f"</div>"
            f"<div style='font-size:0.75rem;color:#888;margin:5px 0 3px;'>1. Eğitim (Ham Veri)</div>"
            f"<div style='background:#e5e7eb;border-radius:999px;height:9px;overflow:hidden;'>"
            f"<div style='width:{ham}%;height:100%;background:#9ca3af;border-radius:999px;'></div></div>"
            f"<div style='font-size:0.75rem;color:#888;margin:5px 0 3px;'>2. Eğitim (Temiz Veri)</div>"
            f"<div style='background:#ede9fe;border-radius:999px;height:9px;overflow:hidden;'>"
            f"<div style='width:{temiz}%;height:100%;background:#7c3aed;border-radius:999px;'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

with col_g:
    grafik = pd.DataFrame(
        {
            "1. Eğitim (Ham)":   [m[1] for m in metrikler],
            "2. Eğitim (Temiz)": [m[2] for m in metrikler],
        },
        index=[m[0] for m in metrikler],
    )
    st.bar_chart(grafik, height=270, color=["#9ca3af", "#7c3aed"])

    theme.tip(
        "<b>Temel Bulgu:</b> Veri temizliği tek başına <b>+%19.1 mAP</b> artışı sağladı. "
        "Model mimarisi değil, <b>veri kalitesi</b> belirleyici oldu.",
        kind="ok",
    )

# ── Veri Kalitesi Analizi ────────────────────────────────────────────────────
theme.section("Veri Kalitesi Analizi", "Eğitim öncesi tespit edilen sorunlar")

col_s, col_t = st.columns(2)

with col_s:
    sorunlar = [
        ("Çok küçük bbox (alan < %0.1)", 1810, "#dc2626"),
        ("Görüntü sınırlarını aşan bbox", 109,  "#d97706"),
        ("Aşırı en-boy oranı (>10x)",     3,    "#ca8a04"),
        ("Tekrar eden bbox (IoU > 0.80)",  3,    "#ca8a04"),
    ]
    for sorun, adet, clr in sorunlar:
        st.markdown(
            f"<div style='background:#ffffff;border-left:4px solid {clr};"
            f"border-radius:0 10px 10px 0;padding:10px 14px;margin-bottom:7px;"
            f"box-shadow:0 1px 2px rgba(0,0,0,0.05);'>"
            f"<div style='font-size:0.83rem;color:#444;'>{sorun}</div>"
            f"<div style='font-weight:700;font-size:1rem;color:{clr};'>{adet:,} adet</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

with col_t:
    temizlik = [
        ("bos_arazi etiketi kaldırıldı", "39.119", "Modelin dikkatini dağıtıyordu"),
        ("Küçük/gürültülü bbox silindi",  "8.434",  "Alan < %0.1 piksel"),
        ("Sınıf sayısı",                  "4 → 3",  "bos_arazi çıkarıldı"),
    ]
    for etiket, deger, aciklama in temizlik:
        st.markdown(
            f"<div style='background:#f5f3ff;border:1px solid #ece9fb;"
            f"border-radius:10px;padding:12px 16px;margin-bottom:8px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-size:0.83rem;color:#4c1d95;font-weight:600;'>{etiket}</span>"
            f"<span style='font-weight:700;font-size:1rem;color:#7c3aed;'>{deger}</span>"
            f"</div>"
            f"<div style='font-size:0.75rem;color:#6d28d9;margin-top:3px;'>{aciklama}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    theme.tip(
        "Bu temizlik işlemi, modelin yanlış sınıfları öğrenmesini engellemiş "
        "ve hem precision hem recall'u eş zamanlı iyileştirmiştir.",
        kind="info",
    )

# ── Model Teknik Detayları ───────────────────────────────────────────────────
theme.section("Model Teknik Detayları")

c1, c2, c3 = st.columns(3)
detaylar = [
    [("Model", "YOLOv11n"), ("Parametre Sayısı", "~2.6M"), ("Model Boyutu", "5.5 MB")],
    [("Epoch", "94 / 100"), ("Batch Size", "16"), ("Görüntü Boyutu", "640×640 px")],
    [("Optimizer", "AdamW"), ("Öğrenme Oranı", "0.001429"), ("GPU", "Tesla T4 — 15 GB")],
]
for col, satirlar in zip([c1, c2, c3], detaylar):
    html = ("<div style='background:#ffffff;border:1px solid #ece9fb;border-radius:12px;"
            "padding:14px 16px;box-shadow:0 1px 2px rgba(124,58,237,0.05);'>")
    for k, v in satirlar:
        html += (
            f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
            f"border-bottom:1px solid #f5f3ff;font-size:0.84rem;'>"
            f"<span style='color:#6d28d9;'>{k}</span>"
            f"<span style='font-weight:600;color:#2e1065;'>{v}</span></div>"
        )
    html += "</div>"
    col.markdown(html, unsafe_allow_html=True)

st.divider()
st.caption("YMH455 Bitirme Projesi · Mustafa BAL (220541003) · 2026")
