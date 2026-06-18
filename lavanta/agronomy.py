"""
Preskriptif zirai karar destek (reçeteleme) motoru.

`omer/.../frontend/src/pages/Analysis.jsx` içindeki "Akıllı Zirai Reçeteleme"
ve hasat hazırlık önerileri Python'a taşınmıştır. Tespit + ekonomi sonuçlarından
çiftçiye yönelik Türkçe aksiyon önerileri üretir.
"""

from lavanta import config


def harvest_advice(readiness: float) -> tuple[str, str]:
    """Hasat hazırlık oranına göre (mesaj, kind) döner. kind: ok | warn."""
    if readiness >= 90:
        return ("Hasat için ideal dönem! Temmuz 4-15 aralığını ve sabah saatlerini "
                "(06:00-09:00) değerlendirin.", "ok")
    if readiness >= 80:
        return ("Hasat eşiğine ulaşıldı. Sıcaklık ve rüzgâr uygunsa sabah erken "
                "saatlerde biçim planlanabilir.", "ok")
    return ("Tarla henüz tam hasat seviyesine ulaşmadı. Uçucu yağ birikimi için "
            "5-7 gün daha olgunlaşma beklenmesi ve NDVI takibi tavsiye edilir.", "warn")


def build_recommendations(econ: dict, density: dict | None = None) -> list[dict]:
    """
    Reçete satırlarını döner: her biri {tag, text, kind}.
    kind: ok | warn | info  (theme.tip ile uyumlu).
    """
    readiness = econ["harvest_readiness"]
    wild_ratio = econ.get("wild_ratio", 0.0)
    recs: list[dict] = []

    # 1) Rekolte analizi
    recs.append({
        "tag": "ANALİZ",
        "kind": "info",
        "text": (f"Toplam yaş çiçek verim tahmini <b>{econ['estimated_yield']:.1f} kg</b>, "
                 f"uçucu yağ potansiyeli <b>{econ['oil_estimate']:.2f} kg</b> "
                 f"(≈ <b>{econ['revenue_euro']:.0f} €</b> gelir)."),
    })

    # 2) Yabani ot / sağlık
    if wild_ratio >= 15:
        recs.append({
            "tag": "SAĞLIK", "kind": "warn",
            "text": (f"Yabani ot oranı <b>%{wild_ratio:.1f}</b> — kritik seviyede. "
                     f"Mekanik çapalama veya seçici herbisit uygulaması önerilir."),
        })
    else:
        recs.append({
            "tag": "SAĞLIK", "kind": "ok",
            "text": (f"Yabani ot oranı <b>%{wild_ratio:.1f}</b>. Tarla genelinde kritik "
                     f"yabani ot istilası bulunmamaktadır."),
        })

    # 3) Yoğunluk (varsa)
    if density is not None:
        ratio = density.get("ratio_pct", 0)
        if ratio < 10:
            recs.append({
                "tag": "YOĞUNLUK", "kind": "warn",
                "text": (f"Bitki yoğunluğu <b>%{ratio:.1f}</b> ile düşük. Boş bölgelerde "
                         f"fide tamamlama (gap-filling) değerlendirilmelidir."),
            })
        else:
            recs.append({
                "tag": "YOĞUNLUK", "kind": "info",
                "text": f"Tarla bitki yoğunluğu <b>%{ratio:.1f}</b> seviyesinde.",
            })

    # 4) Eylem / hasat
    msg, kind = harvest_advice(readiness)
    recs.append({
        "tag": "EYLEM", "kind": kind,
        "text": f"Hasat hazırlığı <b>%{readiness}</b>. {msg}",
    })

    return recs


def recommendations_text(econ: dict, density: dict | None = None) -> str:
    """Reçeteyi düz metin (PDF/indir için) olarak döner."""
    lines = ["LAVANTA ZİRAİ KARAR DESTEK REÇETESİ", "=" * 40, ""]
    for r in build_recommendations(econ, density):
        clean = r["text"].replace("<b>", "").replace("</b>", "")
        lines.append(f"[{r['tag']}] {clean}")
    lines += ["", "Doz kademeleri (azot, kg/da):"]
    for k, v in config.VRA_DOSES.items():
        lines.append(f"  - {k.upper()} yoğun bölge: {v} kg/da")
    return "\n".join(lines)
