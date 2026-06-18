# -*- coding: utf-8 -*-
"""
Veri Seti — GERÇEK istatistikler.

train/valid/test klasörlerindeki gerçek görüntü sayıları ve etiket
dosyalarından hesaplanan sınıf dağılımı.
"""

import pandas as pd
import streamlit as st

from lavanta import theme, refdata, config

theme.inject()

st.title("Veri Seti")
st.caption("Roboflow lavanta veri seti — gerçek sayımlar")

stats = refdata.dataset_stats()
splits = stats["splits"]

# ── Görüntü sayıları ─────────────────────────────────────────────────────────
theme.section("Görüntü Dağılımı")
k = st.columns(4)
k[0].metric("Toplam Görüntü", f"{stats['total_images']:,}")
k[1].metric("Train", f"{splits['train']:,}")
k[2].metric("Valid", f"{splits['valid']:,}")
k[3].metric("Test", f"{splits['test']:,}")

st.bar_chart(pd.Series(splits, name="Görüntü").rename_axis("Split"), height=240)

# ── Sınıf dağılımı ───────────────────────────────────────────────────────────
theme.section("Sınıf Dağılımı (etiket örneği sayısı)")
inst = stats["instances"]
if inst:
    labels = {name: config.CLASSES.get(name, {}).get("label", name) for name in inst}
    series = pd.Series({labels[n]: c for n, c in inst.items()}, name="Örnek")
    cc1, cc2 = st.columns([2, 1])
    cc1.bar_chart(series.rename_axis("Sınıf"), height=260)
    with cc2:
        for name, c in inst.items():
            st.metric(config.CLASSES.get(name, {}).get("label", name), f"{c:,}")
else:
    st.info("Etiket dosyaları bulunamadı.")

# ── Kaynak & augmentasyon ────────────────────────────────────────────────────
theme.section("Kaynak ve Augmentasyon")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Kaynak**")
    st.write(stats["source"])
    st.markdown("**Sınıflar**")
    st.write(", ".join(config.CLASSES[n]["label"] for n in ("olgunlasmis", "yetismemis")))
with c2:
    st.markdown("**Augmentasyon Teknikleri**")
    for t in stats["augmentation"]:
        st.markdown(f"- {t}")

if config.DATA_YAML.exists():
    with st.expander("data_fixed.yaml"):
        st.code(config.DATA_YAML.read_text(encoding="utf-8"), language="yaml")
