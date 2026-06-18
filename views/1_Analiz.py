# -*- coding: utf-8 -*-
"""
Analiz sayfası — birleşik ana ekran.

Kök app.py'nin tespit deneyimi (YOLO + renk bazlı sınıflandırma + mor maske)
korunur; üzerine Ömer projesinin gerçek özellikleri eklenir:
rekolte/uçucu yağ/gelir, hasat hazırlık, HSV yoğunluk, zirai reçete, VRA GeoJSON
ve analiz geçmişine kayıt.
"""

import base64
import time
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from color_classify import (
    PURPLE_H_LOW, PURPLE_H_HIGH, PURPLE_S_MIN, PURPLE_V_MIN, PURPLE_V_MAX,
    DEFAULT_THRESHOLD,
)
from lavanta import config, theme, detection, density as density_mod, economics, agronomy, vra, history

theme.inject()

st.title("Lavanta Olgunluk, Yabancı Bitki & Rekolte Analizi")
st.caption("YOLOv11 tespiti • olgunlaşmış / yabani ot / yetişmemiş • rekolte / uçucu yağ / gelir tahmini")

# ── Model kontrolü ───────────────────────────────────────────────────────────
if not config.MODEL_PATH.exists():
    st.error(f"Model bulunamadı: `{config.MODEL_PATH}` — önce `train.py` çalıştırın.")
    st.stop()

model = detection.load_model()
model_names = detection.model_class_names(model)
has_wild = any(config.normalize_class(n) == "yabani_bitki" for n in model_names.values())

# ── Sidebar: sınıflandırma modu ──────────────────────────────────────────────
st.sidebar.markdown("## Sınıflandırma Modu")
use_color = st.sidebar.toggle(
    "Renk bazlı sınıflandırma",
    value=False,
    help="Açıkken YOLO bbox konumunu kullanır, sınıf kararı mor piksel oranına göre verilir.",
)

color_threshold = st.sidebar.slider(
    "Olgunluk rengi eşiği (%)",
    min_value=0, max_value=25, value=int(DEFAULT_THRESHOLD * 100), step=1,
    help="YOLO kutusu içindeki mor piksel oranı. Önerilen aralık: %8-%12.",
) / 100.0
show_mask = False
hsv_params: dict = {}
density_hsv_params = {
    "h_low": density_mod.DEFAULT_H_LOW,
    "h_high": density_mod.DEFAULT_H_HIGH,
    "s_min": density_mod.DEFAULT_S_MIN,
    "v_min": density_mod.DEFAULT_V_MIN,
    "v_max": density_mod.DEFAULT_V_MAX,
}
if use_color:
    show_mask = st.sidebar.checkbox("Mor maske overlay göster", value=False)
    with st.sidebar.expander("Gelişmiş HSV Ayarları", expanded=False):
        h_low  = st.slider("H alt",  0,  60, PURPLE_H_LOW)
        h_high = st.slider("H üst", 60, 120, PURPLE_H_HIGH)
        s_min  = st.slider("S min",  0,  60, PURPLE_S_MIN)
        v_min  = st.slider("V min",  0,  60, PURPLE_V_MIN)
        v_max  = st.slider("V max", 150, 255, PURPLE_V_MAX)
        hsv_params = dict(h_low=h_low, h_high=h_high, s_min=s_min, v_min=v_min, v_max=v_max)

with st.sidebar.expander("Tarla Yoğunluk HSV Ayarları", expanded=False):
    density_hsv_params = {
        "h_low": st.slider("Yoğunluk H alt", 90, 150, density_mod.DEFAULT_H_LOW),
        "h_high": st.slider("Yoğunluk H üst", 120, 179, density_mod.DEFAULT_H_HIGH),
        "s_min": st.slider("Yoğunluk S min", 0, 255, density_mod.DEFAULT_S_MIN),
        "v_min": st.slider("Yoğunluk V min", 0, 255, density_mod.DEFAULT_V_MIN),
        "v_max": st.slider("Yoğunluk V max", 80, 255, density_mod.DEFAULT_V_MAX),
    }

st.sidebar.divider()

# ── Görüntü yükle ────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Drone görüntüsü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Analiz için sol panelden ya da yukarıdan bir görüntü yükleyin.")
    st.sidebar.markdown("## Tespit Özeti")
    st.sidebar.caption("Görüntü bekleniyor...")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_np  = np.array(pil_img)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# ── Inference: yalnızca yeni görüntüde çalışır ───────────────────────────────
file_id = uploaded.file_id
if st.session_state.get("analiz_file_id") != file_id:
    with st.spinner("Model çalıştırılıyor..."):
        t0 = time.time()
        st.session_state["analiz_yolo_result"] = detection.run_inference(model, img_bgr)
        st.session_state["analiz_proc_time"] = round(time.time() - t0, 2)
        st.session_state["analiz_img_bgr"] = img_bgr
        st.session_state["analiz_file_id"] = file_id
        st.session_state["analiz_saved_id"] = None  # yeni dosya -> tekrar kaydedilebilir

yolo_result = st.session_state["analiz_yolo_result"]
img_bgr     = st.session_state["analiz_img_bgr"]
proc_time   = st.session_state["analiz_proc_time"]
dens        = density_mod.field_density(img_bgr, **density_hsv_params)


def _jpeg_data_uri(image_rgb: np.ndarray) -> str:
    """RGB görüntüyü karşılaştırma bileşeninde kullanılacak JPEG URI'ına çevirir."""
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not ok:
        raise ValueError("Yoğunluk önizleme görüntüsü kodlanamadı.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def render_density_compare(image_bgr: np.ndarray, overlay_rgb: np.ndarray) -> None:
    """Ham görüntü ile yoğunluk overlay'ini sürüklenebilir yüzde ayarıyla karşılaştırır."""
    raw_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raw_uri = _jpeg_data_uri(raw_rgb)
    overlay_uri = _jpeg_data_uri(overlay_rgb)
    height, width = raw_rgb.shape[:2]
    component_height = max(300, min(720, int(700 * height / max(width, 1)) + 70))

    components.html(
        f"""
        <style>
          * {{ box-sizing: border-box; }}
          body {{ margin: 0; font-family: Arial, sans-serif; color: #4c1d95; }}
          .compare {{
            position: relative; width: 100%; aspect-ratio: {width} / {height};
            overflow: hidden; border: 1px solid #ddd6fe; border-radius: 12px;
            background: #111827; user-select: none;
          }}
          .compare img {{
            position: absolute; inset: 0; width: 100%; height: 100%;
            object-fit: contain; pointer-events: none;
          }}
          #density-overlay {{ clip-path: inset(0 50% 0 0); }}
          #density-divider {{
            position: absolute; top: 0; bottom: 0; left: 50%; width: 3px;
            transform: translateX(-50%); background: #fbbf24;
            box-shadow: 0 0 0 1px rgba(0,0,0,.25); pointer-events: none;
          }}
          #density-divider::after {{
            content: "↔"; position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%); width: 34px; height: 34px;
            display: grid; place-items: center; border-radius: 50%;
            color: #1f2937; background: #fbbf24; border: 2px solid #fff;
            font-weight: 700; box-shadow: 0 2px 8px rgba(0,0,0,.35);
          }}
          #density-slider {{
            position: absolute; inset: 0; width: 100%; height: 100%;
            margin: 0; opacity: 0; cursor: ew-resize;
          }}
          .labels {{
            display: flex; justify-content: space-between; align-items: center;
            gap: 8px; margin-top: 8px; font-size: 12px; font-weight: 700;
          }}
          #density-value {{ color: #6d28d9; white-space: nowrap; }}
        </style>
        <div class="compare">
          <img src="{raw_uri}" alt="Ham drone görüntüsü">
          <img id="density-overlay" src="{overlay_uri}" alt="Lavanta yoğunluk overlay görüntüsü">
          <div id="density-divider"></div>
          <input id="density-slider" type="range" min="0" max="100" value="50"
                 aria-label="Yoğunluk karşılaştırma yüzdesi">
        </div>
        <div class="labels">
          <span>AI YOĞUNLUK OVERLAY</span>
          <span id="density-value">%50</span>
          <span>HAM GÖRÜNTÜ</span>
        </div>
        <script>
          const slider = document.getElementById("density-slider");
          const overlay = document.getElementById("density-overlay");
          const divider = document.getElementById("density-divider");
          const value = document.getElementById("density-value");
          slider.addEventListener("input", () => {{
            const position = Number(slider.value);
            overlay.style.clipPath = `inset(0 ${{100 - position}}% 0 0)`;
            divider.style.left = `${{position}}%`;
            value.textContent = `%${{position}}`;
          }});
        </script>
        """,
        height=component_height,
        scrolling=False,
    )

# ── Filtre + çizim (slider değişiminde anında yeniden çalışır) ────────────────
detections = detection.filter_boxes(
    yolo_result, model_names,
    image_bgr=img_bgr if use_color else None,
    use_color=use_color, color_threshold=color_threshold, hsv_params=hsv_params,
)
counts    = detection.count_classes(detections)
econ      = economics.calculate_yield(counts)
annotated = detection.draw_boxes(img_bgr, detections, show_purple_pct=use_color)
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ── Geçmişe kaydet (yeni dosya için bir kez) ─────────────────────────────────
if st.session_state.get("analiz_saved_id") is None:
    record = {
        "filename": uploaded.name,
        "density_score": dens["density"],
        "processing_time": proc_time,
        **econ,
    }
    saved = history.save_analysis(record)
    st.session_state["analiz_saved_id"] = saved["id"]

# ── Sidebar: tespit özeti ────────────────────────────────────────────────────
st.sidebar.markdown("## Tespit Özeti")
c1, c2 = st.sidebar.columns(2)
c1.metric("Olgunlaşmış", counts["olgunlasmis"])
c2.metric("Yetişmemiş", counts["yetismemis"])
st.sidebar.metric("Yabani Ot 🌿", counts.get("yabani_bitki", 0))
st.sidebar.markdown(f"**Toplam tespit:** {len(detections)}")
st.sidebar.divider()

for cls_name in config.CLASSES:
    cls_dets = [d for d in detections if d["cls"] == cls_name]
    if not cls_dets:
        continue
    hexc = config.hex_color(cls_name)
    st.sidebar.markdown(
        f"<span style='color:{hexc};font-weight:bold'>■ {config.CLASSES[cls_name]['label']} "
        f"({len(cls_dets)})</span>", unsafe_allow_html=True)
    confs = [d["conf"] for d in cls_dets]
    st.sidebar.caption(f"Ort: {np.mean(confs):.3f} | Min: {min(confs):.3f} | Max: {max(confs):.3f}")

# ── Ana alan: tespit görselleri ──────────────────────────────────────────────
theme.section("Tespit Görselleri", f"{uploaded.name} • {len(detections)} obje • {proc_time}s")

if use_color and show_mask:
    col_o, col_p, col_m = st.columns(3)
else:
    col_o, col_p = st.columns(2)
    col_m = None

with col_o:
    st.subheader("Orijinal")
    st.image(pil_img, width="stretch")
with col_p:
    st.subheader(f"Tespit ({len(detections)})")
    st.image(annotated_rgb, width="stretch")
if col_m is not None:
    mask_overlay = detection.build_mask_overlay(img_bgr, detections, hsv_params)
    with col_m:
        st.subheader("Mor Maske")
        st.image(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB), width="stretch")

st.download_button(
    "Tespit sonucunu indir (JPG)",
    data=cv2.imencode(".jpg", annotated)[1].tobytes(),
    file_name=f"tespit_{uploaded.name}",
    mime="image/jpeg",
)

# ── Rekolte & ekonomi ────────────────────────────────────────────────────────
theme.section("Rekolte & Ekonomik Tahmin", "3. yıl referansı • %2 randıman • 80 €/kg")

k = st.columns(3)
k[0].metric("Olgun Bitki", econ["mature_count"])
k[1].metric("Yetişmemiş", econ["immature_count"])
k[2].metric("Yabani Ot", econ["wild_count"])

k = st.columns(3)
k[0].metric("Tahmini Yaş Çiçek", f"{econ['estimated_yield']:.1f} kg")
k[1].metric("Uçucu Yağ", f"{econ['oil_estimate']:.2f} kg")
k[2].metric("Tahmini Gelir", f"{econ['revenue_euro']:.0f} €")

# Hasat hazırlık bar
readiness = econ["harvest_readiness"]
bar_color = "#16a34a" if readiness >= 80 else "#d97706"
st.markdown(
    f"""
    <div style="margin-top:0.6rem">
      <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#4c1d95;font-weight:600">
        <span>Hasat Hazırlık Oranı</span><span>%{readiness}</span>
      </div>
      <div style="background:#ede9fe;border-radius:999px;height:12px;overflow:hidden;margin-top:4px">
        <div style="width:{readiness}%;height:100%;background:{bar_color};border-radius:999px"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tarla yoğunluk haritası ──────────────────────────────────────────────────
theme.section("Tarla Yoğunluk Haritası (HSV Segmentasyon)",
              "Tüm görüntüdeki mor (lavanta) piksel yoğunluğu")
dc1, dc2 = st.columns([1, 2])
dc1.metric("Lavanta Yoğunluğu", f"%{dens['ratio_pct']:.1f}")
dc1.caption(f"{dens['lavender_pixels']:,} mor piksel")
if dens["lavender_pixels"] == 0:
    dc1.warning("Seçili HSV aralığında lavanta pikseli bulunamadı. Yoğunluk HSV ayarlarını genişletin.")
dc1.caption(
    f"H:{dens['params']['h_low']}-{dens['params']['h_high']}  "
    f"S≥{dens['params']['s_min']}  V:{dens['params']['v_min']}-{dens['params']['v_max']}"
)

with dc2:
    render_density_compare(img_bgr, dens["overlay_rgb"])
    st.caption("Ayırıcıyı sürükleyerek ham görüntü ile tarla yoğunluk overlay'ini karşılaştırın.")

# ── Zirai karar destek (reçete) ──────────────────────────────────────────────
theme.section("Akıllı Zirai Karar Destek (Reçete)", "Preskriptif agronomi motoru")
for rec in agronomy.build_recommendations(econ, dens):
    theme.tip(f"<b>[{rec['tag']}]</b> {rec['text']}", kind=rec["kind"])

dl1, dl2 = st.columns(2)
dl1.download_button(
    "🛰️ Traktör Reçetesini İndir (GeoJSON / VRA)",
    data=vra.geojson_bytes(econ),
    file_name=f"VRA_recete_{uploaded.name.split('.')[0]}.geojson",
    mime="application/geo+json",
    width="stretch",
)
dl2.download_button(
    "📄 Zirai Reçete Raporunu İndir (TXT)",
    data=agronomy.recommendations_text(econ, dens).encode("utf-8"),
    file_name=f"recete_{uploaded.name.split('.')[0]}.txt",
    mime="text/plain",
    width="stretch",
)
