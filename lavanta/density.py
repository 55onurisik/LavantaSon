"""
Tarla geneli lavanta yoğunluğu (HSV segmentasyon).

`omer/.../backend/main.py` içindeki opencv_preprocess mantığını tüm görüntüye
uygular. HSV sınırları, bu drone veri setinde doğrulanmış bbox-içi renk
sınıflandırıcısıyla ortak kullanılır; böylece olgunluk ve tarla yoğunluğu aynı
renk kalibrasyonuna göre hesaplanır.
"""

import cv2
import numpy as np

from color_classify import (
    PURPLE_H_HIGH,
    PURPLE_H_LOW,
    PURPLE_S_MIN,
    PURPLE_V_MAX,
    PURPLE_V_MIN,
)

# Tek kalibrasyon kaynağı: renk sınıflandırıcısında %93 doğrulukla kullanılan
# OpenCV HSV sınırları. Ayrı sabitler zamanla iki analizin sapmasını önler.
DEFAULT_H_LOW = PURPLE_H_LOW
DEFAULT_H_HIGH = PURPLE_H_HIGH
DEFAULT_S_MIN = PURPLE_S_MIN
DEFAULT_V_MIN = PURPLE_V_MIN
DEFAULT_V_MAX = PURPLE_V_MAX


def field_density(
    img_bgr: np.ndarray,
    h_low: int = DEFAULT_H_LOW,
    h_high: int = DEFAULT_H_HIGH,
    s_min: int = DEFAULT_S_MIN,
    v_min: int = DEFAULT_V_MIN,
    v_max: int = DEFAULT_V_MAX,
) -> dict:
    """
    Görüntüdeki mor (lavanta) piksel yoğunluğunu hesaplar.

    Döner:
      density      : 0.0-1.0 arası oran
      ratio_pct    : yüzde (0-100)
      mask_rgb     : MAGMA renkli yoğunluk maskesi (RGB, st.image için)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_high, 255, v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morfolojik temizlik
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    total = mask.shape[0] * mask.shape[1]
    lavender_px = int(cv2.countNonZero(mask))
    density = lavender_px / total if total > 0 else 0.0

    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_MAGMA)

    # Saf siyah maske yerine, kullanıcıya bağlam verecek bir overlay üret.
    base = cv2.addWeighted(img_bgr, 0.75, np.full_like(img_bgr, 24), 0.25, 0)
    overlay_bgr = base.copy()
    mask_any = mask > 0
    blended = (
        img_bgr.astype(np.float32) * 0.35
        + mask_colored.astype(np.float32) * 0.65
    ).clip(0, 255).astype(np.uint8)
    overlay_bgr[mask_any] = blended[mask_any]

    mask_rgb = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return {
        "density": round(density, 4),
        "ratio_pct": round(density * 100, 2),
        "lavender_pixels": lavender_px,
        "mask_rgb": mask_rgb,
        "overlay_rgb": overlay_rgb,
        "params": {
            "h_low": h_low,
            "h_high": h_high,
            "s_min": s_min,
            "v_min": v_min,
            "v_max": v_max,
        },
    }
