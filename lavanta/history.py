"""
Analiz geçmişi — SQLite kalıcılığı.

`omer/.../backend/database.py` Streamlit'e uyarlanmıştır. Her analiz
`data/lavanta.db` içine kaydedilir; Panel ve Geçmiş sayfaları buradan okur.
"""

import json
import sqlite3
from datetime import datetime

from lavanta import config


def _connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tabloyu oluşturur (idempotent)."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id       TEXT    UNIQUE NOT NULL,
            filename          TEXT    NOT NULL,
            date              TEXT    NOT NULL,
            mature_count      INTEGER DEFAULT 0,
            immature_count    INTEGER DEFAULT 0,
            wild_count        INTEGER DEFAULT 0,
            estimated_yield   REAL    DEFAULT 0,
            oil_estimate      REAL    DEFAULT 0,
            revenue_euro      REAL    DEFAULT 0,
            density_score     REAL    DEFAULT 0,
            harvest_readiness REAL    DEFAULT 0,
            processing_time   REAL    DEFAULT 0,
            raw_result        TEXT,
            created_at        TEXT    DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def save_analysis(data: dict) -> dict:
    """Bir analiz sonucunu kaydeder; analysis_id ve date ile zenginleştirir."""
    init_db()
    conn = _connect()
    analysis_id = data.get("id") or f"ANL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    conn.execute("""
        INSERT OR REPLACE INTO analyses
        (analysis_id, filename, date, mature_count, immature_count, wild_count,
         estimated_yield, oil_estimate, revenue_euro, density_score,
         harvest_readiness, processing_time, raw_result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_id,
        data.get("filename", "bilinmeyen"),
        now,
        data.get("mature_count", 0),
        data.get("immature_count", 0),
        data.get("wild_count", 0),
        data.get("estimated_yield", 0),
        data.get("oil_estimate", 0),
        data.get("revenue_euro", 0),
        data.get("density_score", 0),
        data.get("harvest_readiness", 0),
        data.get("processing_time", 0),
        json.dumps(data, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()
    data["id"] = analysis_id
    data["date"] = now
    return data


def get_all_analyses(limit: int = 200) -> list[dict]:
    """Tüm analizleri yeniden eskiye döner."""
    init_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r["analysis_id"],
            "filename": r["filename"],
            "date": r["date"],
            "mature_count": r["mature_count"],
            "immature_count": r["immature_count"],
            "wild_count": r["wild_count"],
            "estimated_yield": r["estimated_yield"],
            "oil_estimate": r["oil_estimate"],
            "revenue_euro": r["revenue_euro"],
            "density_score": r["density_score"],
            "harvest_readiness": r["harvest_readiness"],
            "processing_time": r["processing_time"],
        })
    return out


def get_analysis_count() -> int:
    init_db()
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    conn.close()
    return int(n)


def clear_history():
    """Tüm geçmişi siler."""
    init_db()
    conn = _connect()
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()
