"""
Gerçek model & veri seti verileri.

Ömer'in sabit (mock) metrikleri yerine, bizim eğitilmiş modelimizin GERÇEK
çıktıları kullanılır:
  - runs/lavanta_yolo11/results.csv  (eğitim eğrileri, final metrikler)
  - runs/lavanta_yolo11/*.png        (confusion matrix, PR/F1 eğrileri)
  - train|valid|test/labels          (sınıf dağılımı)
"""

import pandas as pd
import streamlit as st

from lavanta import config


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame | None:
    """results.csv'i temiz kolon isimleriyle yükler. Yoksa None."""
    if not config.RESULTS_CSV.exists():
        return None
    df = pd.read_csv(config.RESULTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df


def model_summary() -> dict | None:
    """Son epoch'un metriklerini ve eğitim bilgisini döner."""
    df = load_results()
    if df is None or df.empty:
        return None
    last = df.iloc[-1]

    def g(col, default=0.0):
        return float(last[col]) if col in df.columns else default

    return {
        "epochs": int(last["epoch"]) if "epoch" in df.columns else len(df),
        "precision": g("metrics/precision(B)"),
        "recall": g("metrics/recall(B)"),
        "map50": g("metrics/mAP50(B)"),
        "map50_95": g("metrics/mAP50-95(B)"),
        "train_box_loss": g("train/box_loss"),
        "val_box_loss": g("val/box_loss"),
    }


def metric_curves() -> pd.DataFrame | None:
    """epoch indeksli precision/recall/mAP eğrileri."""
    df = load_results()
    if df is None:
        return None
    cols = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@50",
        "metrics/mAP50-95(B)": "mAP@50-95",
    }
    have = {k: v for k, v in cols.items() if k in df.columns}
    out = df[["epoch", *have.keys()]].rename(columns=have).set_index("epoch")
    return out


def loss_curves() -> pd.DataFrame | None:
    """epoch indeksli eğitim/doğrulama kayıp eğrileri."""
    df = load_results()
    if df is None:
        return None
    cols = {
        "train/box_loss": "train box",
        "train/cls_loss": "train cls",
        "val/box_loss": "val box",
        "val/cls_loss": "val cls",
    }
    have = {k: v for k, v in cols.items() if k in df.columns}
    out = df[["epoch", *have.keys()]].rename(columns=have).set_index("epoch")
    return out


def run_image(name: str):
    """runs/lavanta_yolo11 altındaki bir görsel dosyasının yolunu döner (varsa)."""
    p = config.RUN_DIR / name
    return p if p.exists() else None


# ── Veri seti ────────────────────────────────────────────────────────────────

def _count_images(images_dir) -> int:
    if not images_dir.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in exts)


def _count_instances(labels_dir) -> dict:
    """YOLO etiket dosyalarından sınıf-id -> örnek sayısı."""
    counts: dict[int, int] = {}
    if not labels_dir.exists():
        return counts
    for txt in labels_dir.glob("*.txt"):
        try:
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cid = int(float(line.split()[0]))
                counts[cid] = counts.get(cid, 0) + 1
        except (ValueError, OSError):
            continue
    return counts


@st.cache_data(show_spinner=False)
def dataset_stats() -> dict:
    """Gerçek veri seti istatistikleri (görüntü sayıları + sınıf dağılımı)."""
    splits = {
        "train": _count_images(config.TRAIN_DIR / "images"),
        "valid": _count_images(config.VALID_DIR / "images"),
        "test":  _count_images(config.TEST_DIR / "images"),
    }

    # Sınıf id -> kanonik isim (2 sınıflı modelimiz: 0=olgunlasmis, 1=yetismemis)
    id_to_name = {0: "olgunlasmis", 1: "yetismemis", 2: "yabani_bitki"}
    instances: dict[str, int] = {}
    for split, base in (("train", config.TRAIN_DIR), ("valid", config.VALID_DIR), ("test", config.TEST_DIR)):
        for cid, n in _count_instances(base / "labels").items():
            name = id_to_name.get(cid, str(cid))
            instances[name] = instances.get(name, 0) + n

    return {
        "splits": splits,
        "total_images": sum(splits.values()),
        "instances": instances,
        "augmentation": [
            "Yatay / Dikey Çevirme", "±15° Döndürme",
            "Parlaklık (±20%)", "Mozaik", "Gaussian Blur",
        ],
        "source": "DJI drone ile çekilmiş yüksek çözünürlüklü hava fotoğrafları (Roboflow)",
    }
