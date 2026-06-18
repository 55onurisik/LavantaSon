"""
Rekolte / uçucu yağ / gelir hesabı.

Formüller `omer/.../backend/main.py` (calculate_yield) ve `bitirme (1).py`
(run_lavender_pipeline) ile birebir aynıdır.
"""

from lavanta import config


def calculate_yield(counts: dict) -> dict:
    """
    Sınıf sayımlarından (kanonik isim -> adet) tarımsal ve ekonomik tahmin üretir.

    - Yaş çiçek verimi  = olgun bitki * 0.25 kg
    - Uçucu yağ         = yaş çiçek * %2 randıman
    - Gelir             = yağ * 80 €/kg
    - Hasat hazırlığı   = olgun / (olgun + yetişmemiş) * 100
    """
    mature   = int(counts.get("olgunlasmis", 0))
    immature = int(counts.get("yetismemis", 0))
    wild     = int(counts.get("yabani_bitki", 0))
    total_lavender = mature + immature

    wet_flower = mature * config.YIELD_PER_PLANT
    oil        = wet_flower * config.WET_TO_OIL_RATIO
    revenue    = oil * config.OIL_PRICE_EURO

    harvest_readiness = (mature / total_lavender * 100) if total_lavender > 0 else 0.0
    wild_ratio = (wild / (total_lavender + wild) * 100) if (total_lavender + wild) > 0 else 0.0

    return {
        "mature_count": mature,
        "immature_count": immature,
        "wild_count": wild,
        "total_lavender": total_lavender,
        "estimated_yield": round(wet_flower, 2),   # kg (yaş çiçek)
        "oil_estimate": round(oil, 3),             # kg uçucu yağ
        "revenue_euro": round(revenue, 2),         # €
        "harvest_readiness": round(harvest_readiness, 1),  # %
        "wild_ratio": round(wild_ratio, 1),        # %
    }
