"""
VRA (Variable Rate Application) traktör reçetesi — GeoJSON üretimi.

`omer/.../frontend/src/pages/Analysis.jsx` (downloadVRAGeoJSON) Python'a
taşınmıştır. Koordinatlar demo amaçlıdır (gerçek geolokasyon verisi yoktur);
doz/olgunluk alanları analiz sonucundan türetilir. Çıktı, akıllı traktör GPS
ekranlarının okuyabildiği standart CRS84 GeoJSON formatındadır.
"""

import json
from lavanta import config

# Demo parsel koordinatları (Burdur/Isparta lavanta bölgesi civarı, örnek)
_ZONES = [
    {"id": 1, "name": "Kuzey Parseli",  "level": "az",   "alan_m2": 1200,
     "ring": [[30.2410, 37.7540], [30.2415, 37.7540], [30.2415, 37.7545], [30.2410, 37.7545], [30.2410, 37.7540]]},
    {"id": 2, "name": "Merkez Parseli", "level": "orta", "alan_m2": 1800,
     "ring": [[30.2415, 37.7540], [30.2420, 37.7540], [30.2420, 37.7545], [30.2415, 37.7545], [30.2415, 37.7540]]},
    {"id": 3, "name": "Güney Parseli",  "level": "tam",  "alan_m2": 1500,
     "ring": [[30.2420, 37.7540], [30.2425, 37.7540], [30.2425, 37.7545], [30.2420, 37.7545], [30.2420, 37.7540]]},
]

_LEVEL_LABEL = {"az": "Az", "orta": "Orta", "tam": "Tam"}


def build_vra_geojson(econ: dict) -> dict:
    """Analiz sonucundan VRA gübreleme reçetesi (GeoJSON FeatureCollection) üretir."""
    features = []
    for z in _ZONES:
        dose = config.VRA_DOSES[z["level"]]
        features.append({
            "type": "Feature",
            "properties": {
                "BolgeID": z["id"],
                "Isim": z["name"],
                "ReceteDozaj": f"Azot {dose} kg/da",
                "DozKgDa": dose,
                "Olgunluk": _LEVEL_LABEL[z["level"]],
                "AlanM2": z["alan_m2"],
                "HasatHazirlik": econ.get("harvest_readiness", 0),
            },
            "geometry": {"type": "Polygon", "coordinates": [z["ring"]]},
        })

    return {
        "type": "FeatureCollection",
        "name": "Lavanta_Tarla_VRA_Recetesi",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }


def geojson_bytes(econ: dict) -> bytes:
    """İndirilebilir GeoJSON içeriğini bytes olarak döner."""
    data = build_vra_geojson(econ)
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
