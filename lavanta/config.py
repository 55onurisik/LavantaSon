"""
Merkezi sabitler ve yollar.

Tespit sabitleri kök `app.py`'den; ekonomi sabitleri `omer/.../backend/main.py`
ve `bitirme (1).py`'deki Tarım Danışmanı modelinden alınmıştır.
"""

from pathlib import Path

# ── Yollar ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH  = PROJECT_ROOT / "runs" / "mustafa_model" / "best.pt"
RUN_DIR     = PROJECT_ROOT / "runs" / "lavanta_yolo11"
TEST_RUN_DIR = PROJECT_ROOT / "runs" / "lavanta_yolo11_test"
RESULTS_CSV = RUN_DIR / "results.csv"
DATA_YAML   = PROJECT_ROOT / "data_fixed.yaml"

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH  = DATA_DIR / "lavanta.db"

TRAIN_DIR = PROJECT_ROOT / "train"
VALID_DIR = PROJECT_ROOT / "valid"
TEST_DIR  = PROJECT_ROOT / "test"

# ── Sınıflar ─────────────────────────────────────────────────────────────────
# `draw` = OpenCV çizimi için BGR renk (kök app.py ile aynı görünüm).
# Sidebar/HTML için hex, BGR -> RGB çevrilerek `hex_color()` ile üretilir.
CLASSES = {
    "olgunlasmis":  {"label": "Olgunlaşmış", "short": "olgun",     "draw": (34, 197, 94),  "conf": 0.45},
    "yetismemis":   {"label": "Yetişmemiş",  "short": "yetismemis", "draw": (239, 68, 68), "conf": 0.20},
    "yabani_bitki": {"label": "Yabani Ot",   "short": "yabani",     "draw": (0, 140, 255),  "conf": 0.25},
}

# Lavanta (hasat edilebilir) bitki sınıfları — ekonomi/yoğunluk için.
LAVENDER_CLASSES = ("olgunlasmis", "yetismemis")

DEFAULT_CONF = 0.25

# Farklı model sürümlerinden gelen ham sınıf isimlerini standartlaştırır
# (omer/.../backend/main.py içindeki CLASS_NAME_MAP'ten genişletilmiştir).
CLASS_NAME_MAP = {
    "lavanta": "olgunlasmis",
    "olgun": "olgunlasmis",
    "olgunlasmis": "olgunlasmis",
    "olgunlaşmış": "olgunlasmis",
    "yetismemis": "yetismemis",
    "yetişmemiş": "yetismemis",
    "yabani": "yabani_bitki",
    "yabani_bitki": "yabani_bitki",
    "yabani_ot": "yabani_bitki",
    "bos_arazi": "bos_arazi",
}


def normalize_class(raw: str) -> str:
    """Ham sınıf ismini kanonik isme çevirir; eşleşme yoksa kendisini döner."""
    if raw is None:
        return raw
    return CLASS_NAME_MAP.get(raw, CLASS_NAME_MAP.get(str(raw).lower(), raw))


def hex_color(name: str) -> str:
    """Sınıfın sidebar/HTML için hex rengini (RGB) döner."""
    b, g, r = CLASSES[name]["draw"]
    return f"#{r:02x}{g:02x}{b:02x}"


def class_conf(name: str) -> float:
    """Sınıf bazlı confidence eşiği."""
    return CLASSES.get(name, {}).get("conf", DEFAULT_CONF)


# ── Ekonomi sabitleri (Tarım Danışmanı Modeli) ───────────────────────────────
PLANTS_PER_DEKAR         = 2000     # da başına bitki
AVG_WET_FLOWER_PER_DEKAR = 500      # kg, 3. yıl ortalaması (400-650 arası)
WET_TO_OIL_RATIO         = 0.02     # %2 randıman (600 kg yaş -> ~12 kg yağ)
OIL_PRICE_EURO           = 80       # €/kg (60-100 arası ortalama)
YIELD_PER_PLANT          = AVG_WET_FLOWER_PER_DEKAR / PLANTS_PER_DEKAR  # 0.25 kg

# VRA gübreleme reçetesi doz kademeleri (kg/da azot)
VRA_DOSES = {
    "az":   25,   # az yoğun bölge -> yüksek doz
    "orta": 15,
    "tam":  5,    # tam yoğun bölge -> düşük doz
}
