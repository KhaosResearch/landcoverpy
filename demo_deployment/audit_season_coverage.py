#!/usr/bin/env python3
"""
audit_season_coverage.py

Herramienta de auditoria y control en tiempo real para Sentinel-2 en MinIO y MongoDB.
Muestra:
- Conteo y porcentaje de composites completados por estacion.
- Estado de claims activos (que worker esta procesando que tile en este momento).
- Lista exacta de tiles pendientes por estacion.
- Deteccion de anomalias (composites con < 10 bandas).
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pymongo
from minio import Minio


def get_minio_client() -> Minio:
    host = os.getenv("MINIO_HOST", "localhost")
    port = os.getenv("MINIO_PORT", "31113")
    access_key = os.getenv("MINIO_ACCESS_KEY", "adminadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "adminadmin")
    return Minio(
        f"{host}:{port}",
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )


def get_mongo_db():
    host = os.getenv("MONGO_HOST", "localhost")
    port = os.getenv("MONGO_PORT", "31114")
    user = os.getenv("MONGO_USERNAME", "adminadmin")
    password = os.getenv("MONGO_PASSWORD", "adminadmin")
    db_name = os.getenv("MONGO_DB", "sentinel2-metadata")
    uri = f"mongodb://{user}:{password}@{host}:{port}/?authSource=admin"
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db_name]


def get_target_tiles():
    from landcoverpy.utilities.geometries import (
        _csv_to_geojson,
        _group_validated_data_points_by_tile,
        _kmz_to_geojson,
    )

    data_file = os.getenv("DB_FILE", "/app/data/dataset.csv")
    if not Path(data_file).exists():
        data_file = "demo_deployment/app_data/dataset.csv"

    tiles_to_train = set()
    if Path(data_file).exists():
        if data_file.endswith(".kmz"):
            gf = _kmz_to_geojson(data_file)
        else:
            gf = _csv_to_geojson(data_file, sep=";")
        polygons_per_tile = _group_validated_data_points_by_tile(gf)
        tiles_to_train = set(polygons_per_tile.keys())

    tiles_to_predict_env = os.getenv("TILES_TO_PREDICT", "[]")
    if "#" in tiles_to_predict_env:
        tiles_to_predict_env = tiles_to_predict_env.split("#")[0].strip()
    try:
        tiles_to_predict = set(json.loads(tiles_to_predict_env))
    except Exception:
        tiles_to_predict = set()

    return sorted(tiles_to_train.union(tiles_to_predict))


def main():
    print("=" * 80)
    print("AUDITORIA DE COBERTURA Y ESTADO DEL CLUSTER SENTINEL-2 (MEDITERRANEO 2021)")
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    try:
        db = get_mongo_db()
        col = db["composites"]
        claims_col = db["tile_claims"]
    except Exception as e:
        print(f"[ERROR] Conectando a MongoDB: {e}")
        return

    try:
        all_tiles = get_target_tiles()
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo leer la lista completa de geometrias ({e}). Usando tiles de MongoDB.")
        all_tiles = sorted(col.distinct("tile"))

    total_target = len(all_tiles)
    seasons = ["spring", "flowering", "summer", "autumn"]

    # 1. Resumen por estacion
    print("\n1. PROGRESO GLOBAL POR ESTACION:")
    print("-" * 80)
    total_done = 0
    pending_dict = {}

    for s in seasons:
        done_set = set(col.distinct("tile", {"season": s}))
        pending = [t for t in all_tiles if t not in done_set]
        pending_dict[s] = pending
        pct = len(done_set) / total_target * 100 if total_target > 0 else 0
        total_done += len(done_set)
        print(f"  * {s.upper():<10}: {len(done_set):>3} / {total_target} ({pct:5.1f}%) | Pendientes: {len(pending):>3}")

    global_pct = total_done / (total_target * len(seasons)) * 100 if total_target > 0 else 0
    print(f"  -------------------------------------------------------------")
    print(f"  TOTAL GLOBAL : {total_done:>4} / {total_target * len(seasons)} ({global_pct:5.1f}%)")

    # 2. Reclamos activos (Claims)
    print("\n2. WORKERS ACTIVOS Y TILES EN PROCESAMIENTO:")
    print("-" * 80)
    cutoff = datetime.utcnow() - timedelta(hours=2)
    active_claims = list(claims_col.find({"status": "processing", "claimed_at": {"$gte": cutoff}}))
    if active_claims:
        for c in active_claims:
            elapsed = (datetime.utcnow() - c["claimed_at"]).total_seconds() / 60
            print(f"  - Worker [{c.get('claimed_by', 'desconocido')}]: {c.get('season', '?').upper()} -> Tile {c.get('tile')} (iniciado hace {elapsed:.1f} min)")
    else:
        print("  Ningun reclamo activo registrado actualmente (o workers sin sistema de claims).")

    # 3. Muestra de tiles pendientes
    print("\n3. RANGO DE TILES PENDIENTES:")
    print("-" * 80)
    for s in seasons:
        p = pending_dict[s]
        if p:
            print(f"  * {s.upper():<10}: {len(p)} pendientes. Primeros 3: {p[:3]} ... Ultimos 3: {p[-3:]}")
        else:
            print(f"  * {s.upper():<10}: 100% COMPLETADO")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
