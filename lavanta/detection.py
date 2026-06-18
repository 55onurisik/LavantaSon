"""
YOLO tespiti + renk bazlı yeniden sınıflandırma + görselleştirme.

Kök `app.py`'deki mantık buraya taşınmış, kanonik sınıf isimleriyle
(olgunlasmis / yetismemis / yabani_bitki) genelleştirilmiştir.
"""

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from color_classify import purple_ratio, get_purple_mask, DEFAULT_THRESHOLD
from lavanta import config


@st.cache_resource(show_spinner=False)
def load_model(model_path: str | None = None):
    """Modeli yükler (cache'lenir). model_path verilmezse config.MODEL_PATH."""
    path = model_path or str(config.MODEL_PATH)
    return YOLO(str(path))


def model_class_names(model) -> dict:
    """Modelin {id: ham_isim} sözlüğünü döner."""
    return dict(model.names)


def run_inference(model, image_bgr: np.ndarray, conf: float | None = None):
    """Tek görüntü üzerinde YOLO tahmini çalıştırır, ilk sonucu döner."""
    if conf is None:
        conf = min(c["conf"] for c in config.CLASSES.values())
    results = model.predict(source=image_bgr, conf=conf, imgsz=640, verbose=False)
    return results[0]


def filter_boxes(result, model_names: dict, image_bgr=None, use_color=False,
                 color_threshold=DEFAULT_THRESHOLD, hsv_params=None) -> list:
    """
    YOLO kutularını sınıf bazlı confidence eşiğiyle filtreler.

    use_color=True iken olgun/yetişmemiş kararı YOLO yerine mor piksel oranına
    devredilir (yabani_bitki kutuları bundan etkilenmez).
    """
    if hsv_params is None:
        hsv_params = {}

    kept = []
    for box in result.boxes:
        raw_id   = int(box.cls)
        raw_name = model_names.get(raw_id, str(raw_id))
        cls_name = config.normalize_class(raw_name)
        conf     = float(box.conf)

        if conf < config.class_conf(cls_name):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        purple_pct = None
        final_cls  = cls_name

        if use_color and image_bgr is not None and cls_name in config.LAVENDER_CLASSES:
            crop       = image_bgr[y1:y2, x1:x2]
            purple_pct = purple_ratio(crop, **hsv_params)
            final_cls  = "olgunlasmis" if purple_pct >= color_threshold else "yetismemis"

        kept.append({
            "cls": final_cls,
            "yolo_cls": cls_name,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "purple_pct": purple_pct,
        })

    kept.sort(key=lambda d: d["conf"], reverse=True)
    return kept


def count_classes(detections: list) -> dict:
    """Kanonik sınıf -> adet sözlüğü döner (tüm bilinen sınıflar için anahtar içerir)."""
    counts = {name: 0 for name in config.CLASSES}
    for d in detections:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    return counts


def draw_boxes(image_bgr: np.ndarray, detections: list,
               show_purple_pct: bool = False) -> np.ndarray:
    """Tespit kutularını ve etiketlerini görüntüye çizer (BGR döner)."""
    out = image_bgr.copy()
    for d in detections:
        cls_name = d["cls"]
        color = config.CLASSES[cls_name]["draw"]
        x1, y1, x2, y2 = d["bbox"]

        label = f"{config.CLASSES[cls_name]['short']}  {d['conf']:.2f}"
        if show_purple_pct and d.get("purple_pct") is not None:
            label += f"  | mor:{d['purple_pct']*100:.0f}%"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_mask_overlay(image_bgr: np.ndarray, detections: list,
                       hsv_params: dict | None = None) -> np.ndarray:
    """Tüm bbox'ların mor maskelerini %60 alpha ile görüntüye bindirir (BGR döner)."""
    if hsv_params is None:
        hsv_params = {}

    overlay      = image_bgr.copy()
    purple_layer = np.zeros_like(image_bgr)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        crop = image_bgr[y1:y2, x1:x2]
        mask = get_purple_mask(crop, **hsv_params)
        purple_layer[y1:y2, x1:x2][mask > 0] = (180, 0, 180)  # BGR mor

    mask_any = np.any(purple_layer > 0, axis=2)
    blended  = cv2.addWeighted(image_bgr, 0.4, purple_layer, 0.6, 0)
    overlay[mask_any] = blended[mask_any]
    return overlay
