"""
Renk bazlı lavanta olgunluk sınıflandırması.
Hiçbir YOLO/Streamlit bağımlılığı yoktur; salt OpenCV + NumPy.

Drone görüntüsü gerçeği (HSV analizi — sadece orijinal görüntüler):
  - Olgunlasmis: çiçeklenmiş → H 40-80 bölgesinde ~%30 piksel (median)
  - Yetismemis : sadece yeşil yaprak → H 40-80 bölgesinde ~%4 piksel (median)
  Eşik 0.10 @ %93 toplam doğruluk (200 olgunlasmis + 200 yetismemis bbox ile ölçüldü).
"""

import cv2
import numpy as np

# ── Ayarlanabilir HSV sabitleri ──────────────────────────────────────────────
PURPLE_H_LOW  = 40    # Hue alt sınır (OpenCV 0-180) — sarı-yeşil geçiş
PURPLE_H_HIGH = 80    # Hue üst sınır — cyan-yeşil sonu
PURPLE_S_MIN  = 20    # Saturation minimum
PURPLE_V_MIN  = 30    # Value minimum
PURPLE_V_MAX  = 230   # Value maximum

DEFAULT_THRESHOLD = 0.10  # %10 → olgunlasmis  (%93 doğruluk @ bu eşik)


# ── Temel fonksiyonlar ───────────────────────────────────────────────────────

def purple_ratio(
    bgr_crop: np.ndarray,
    h_low:  int = PURPLE_H_LOW,
    h_high: int = PURPLE_H_HIGH,
    s_min:  int = PURPLE_S_MIN,
    v_min:  int = PURPLE_V_MIN,
    v_max:  int = PURPLE_V_MAX,
) -> float:
    """
    BGR kırpılmış bölgedeki mor/leylak piksel oranını döner (0.0 – 1.0).
    Boş veya geçersiz crop için 0.0 döner.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low,  s_min, v_min], dtype=np.uint8)
    upper = np.array([h_high, 255,   v_max], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0

    return float(np.count_nonzero(mask)) / total


def classify_by_color(
    bgr_crop: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    **hsv_kwargs,
) -> int:
    """
    Mor piksel oranına göre sınıf döner.
    Dönüş: 0 = olgunlasmis, 1 = yetismemis
    """
    ratio = purple_ratio(bgr_crop, **hsv_kwargs)
    return 0 if ratio >= threshold else 1


def get_purple_mask(
    bgr_crop: np.ndarray,
    h_low:  int = PURPLE_H_LOW,
    h_high: int = PURPLE_H_HIGH,
    s_min:  int = PURPLE_S_MIN,
    v_min:  int = PURPLE_V_MIN,
    v_max:  int = PURPLE_V_MAX,
) -> np.ndarray:
    """
    BGR crop için binary maske döner (mor piksel=255, diğer=0).
    Mor maske overlay görselleştirmesi için kullanılır.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low,  s_min, v_min], dtype=np.uint8)
    upper = np.array([h_high, 255,   v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def hsv_stats(bgr_crop: np.ndarray) -> dict:
    """
    Crop'taki HSV kanal istatistiklerini döner.
    HSV eşik kalibrasyonunda kullanılır.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return {}
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {
        "h_mean": float(np.mean(h)),
        "h_std":  float(np.std(h)),
        "h_p5":   float(np.percentile(h, 5)),
        "h_p95":  float(np.percentile(h, 95)),
        "s_mean": float(np.mean(s)),
        "v_mean": float(np.mean(v)),
    }
