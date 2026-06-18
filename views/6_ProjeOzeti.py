# -*- coding: utf-8 -*-
"""Lavanta AI uygulamasının güncel proje özeti."""

import streamlit as st

from color_classify import DEFAULT_THRESHOLD
from lavanta import config, density, theme

theme.inject()

st.title("Proje Özeti")
st.caption("Drone görüntülerinden lavanta olgunluğu, yabani ot, tarla yoğunluğu ve rekolte analizi")

theme.section(
    "Projenin Amacı",
    "Tek bir drone görüntüsünden sahada uygulanabilir tarımsal karar desteği üretmek",
)
st.markdown(
    "Lavanta AI; yüksek çözünürlüklü tarla görüntülerini **YOLOv11**, renk tabanlı "
    "olgunluk analizi ve **OpenCV HSV segmentasyonu** ile işler. Sistem yalnızca nesne "
    "tespiti yapmakla kalmaz; olgunluk, yabani ot oranı, tarla yoğunluğu, tahmini yaş "
    "çiçek verimi, uçucu yağ ve gelir sonuçlarını tek analizde birleştirir."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Model", "YOLOv11")
k2.metric("Tespit Sınıfı", len(config.CLASSES))
k3.metric("Olgunluk Eşiği", f"%{DEFAULT_THRESHOLD * 100:.0f}")
k4.metric("Yoğunluk HSV", f"H:{density.DEFAULT_H_LOW}-{density.DEFAULT_H_HIGH}")

theme.section("Sistemin Ürettiği Sonuçlar")
f1, f2 = st.columns(2)

with f1:
    st.markdown("#### Görüntü ve bitki analizi")
    st.markdown(
        "- Olgunlaşmış ve yetişmemiş lavanta tespiti\n"
        "- Yabani ot tespiti ve tarla içindeki oranı\n"
        "- Ayarlanabilir renk yoğunluğu eşiğiyle yeniden sınıflandırma\n"
        "- Kalibre edilmiş HSV aralığıyla tarla yoğunluğu hesabı\n"
        "- Ham görüntü ve mor spektral overlay karşılaştırıcısı"
    )

with f2:
    st.markdown("#### Tarımsal karar desteği")
    st.markdown(
        "- Hasat hazırlık yüzdesi ve zamanlama önerisi\n"
        "- Yaş çiçek, uçucu yağ ve gelir tahmini\n"
        "- Yoğunluk ve yabani ot durumuna göre zirai reçete\n"
        "- Traktör sistemleri için GeoJSON/VRA çıktısı\n"
        "- Analiz geçmişi ve özet paneli"
    )

theme.section(
    "Analiz Akışı",
    "Yüklenen görüntüden raporlanabilir sonuca kadar çalışan işlem hattı",
)

steps = [
    ("1", "Görüntü Girişi", "JPG veya PNG drone görüntüsü RGB/BGR formatına hazırlanır."),
    ("2", "YOLOv11 Tespiti", "Bitkiler üç operasyonel sınıfta konumlandırılır ve sayılır."),
    ("3", "Renk Kalibrasyonu", "Her lavanta kutusundaki renk yoğunluğu seçilen eşikle karşılaştırılır."),
    ("4", "Tarla Yoğunluğu", "Tüm görüntü H:40-80 kalibrasyonuyla segmentlere ayrılır."),
    ("5", "Rekolte ve Ekonomi", "Olgun bitki sayısından yaş çiçek, yağ ve gelir hesaplanır."),
    ("6", "Karar ve Kayıt", "Zirai öneri, VRA çıktısı ve analiz geçmişi oluşturulur."),
]

for number, title, description in steps:
    st.markdown(
        f"<div style='display:flex;gap:12px;align-items:flex-start;background:#fff;"
        f"border:1px solid #ede9fe;border-radius:10px;padding:11px 14px;margin-bottom:7px;'>"
        f"<div style='min-width:28px;height:28px;border-radius:50%;display:grid;place-items:center;"
        f"background:#7c3aed;color:white;font-weight:700;'>{number}</div>"
        f"<div><b style='color:#2e1065'>{title}</b>"
        f"<div style='color:#6b7280;font-size:.84rem;margin-top:2px'>{description}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

theme.section("Tespit Sınıfları ve Karar Mantığı")
c1, c2, c3 = st.columns(3)
class_cards = [
    (c1, "olgunlasmis", "Olgunlaşmış", "Hasat ve rekolte hesabına dahil edilir."),
    (c2, "yetismemis", "Yetişmemiş", "Hasat hazırlık oranını düşürür."),
    (c3, "yabani_bitki", "Yabani Ot", "Tarla sağlığı ve müdahale önerisini etkiler."),
]
for column, name, label, description in class_cards:
    color = config.hex_color(name)
    column.markdown(
        f"<div style='border-top:4px solid {color};background:#fff;border-radius:10px;"
        f"padding:14px;min-height:120px;box-shadow:0 1px 3px rgba(0,0,0,.06)'>"
        f"<div style='font-weight:700;color:{color}'>{label}</div>"
        f"<code>{name}</code>"
        f"<div style='font-size:.82rem;color:#6b7280;margin-top:8px'>{description}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.info(
    "Renk bazlı sınıflandırma açıkken, lavanta kutusundaki kalibre edilmiş renk "
    f"yoğunluğu varsayılan olarak **%{DEFAULT_THRESHOLD * 100:.0f} ve üzerindeyse** "
    "olgunlaşmış; altındaysa yetişmemiş kabul edilir. Eşik analiz ekranından değiştirilebilir."
)

theme.section("Hesaplama Modeli")
h1, h2 = st.columns(2)
with h1:
    st.markdown("#### Rekolte ve ekonomi")
    st.code(
        "Yaş çiçek = olgun bitki × 0.25 kg\n"
        "Uçucu yağ = yaş çiçek × %2\n"
        "Tahmini gelir = uçucu yağ × 80 €/kg",
        language=None,
    )
with h2:
    st.markdown("#### Hasat ve yoğunluk")
    st.code(
        "Hasat hazırlığı = olgun / (olgun + yetişmemiş)\n"
        "Tarla yoğunluğu = seçili HSV pikseli / toplam piksel\n"
        "Varsayılan HSV = H:40-80, S≥20, V:30-230",
        language=None,
    )

theme.section("Teknik Yapı")
t1, t2, t3 = st.columns(3)
model_size = config.MODEL_PATH.stat().st_size / (1024 * 1024) if config.MODEL_PATH.exists() else 0
t1.metric("Model Dosyası", f"{model_size:.1f} MB" if model_size else "Bulunamadı")
t2.metric("Arayüz", "Streamlit")
t3.metric("Kayıt", "SQLite")

st.markdown(
    "- **Görüntü işleme:** OpenCV ve NumPy\n"
    "- **Nesne tespiti:** Ultralytics YOLOv11 / PyTorch\n"
    "- **Renk analizi:** Ayarlanabilir HSV sınırları ve morfolojik temizleme\n"
    "- **Sunum:** Streamlit Cloud üzerinde çok sayfalı uygulama\n"
    "- **Dışa aktarma:** JPEG, TXT ve GeoJSON/VRA"
)

theme.tip(
    "<b>Görselleştirme notu:</b> Mor spektral karşılaştırıcı, görüntü farklarını "
    "incelemek için kullanılan görsel bir katmandır. Sayısal tarla yoğunluğu ayrı olarak "
    "kalibre edilmiş HSV maskesinden hesaplanır.",
    kind="info",
)

st.divider()
st.caption("Lavanta AI · YOLOv11 + renk analizi + tarımsal karar destek sistemi")
