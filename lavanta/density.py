"""
Tarla geneli lavanta yoğunluğu (HSV segmentasyon).

`omer/.../backend/main.py` içindeki opencv_preprocess mantığı: gerçek mor
(H 120-160) piksel oranını hesaplar ve MAGMA renkli bir yoğunluk maskesi üretir.
Bu, kök projedeki bbox-içi renk analizinden (H 40-80) bağımsız, tüm-görüntü
yoğunluk ölçüsüdür.
"""

import cv2
import numpy as np

# Lavanta mor renk aralığı (OpenCV HSV, 0-180 hue)
DEFAULT_H_LOW = 120
DEFAULT_H_HIGH = 160
DEFAULT_S_MIN = 30
DEFAULT_V_MIN = 50
DEFAULT_V_MAX = 255


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
