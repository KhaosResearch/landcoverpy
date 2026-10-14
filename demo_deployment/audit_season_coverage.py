#!/usr/bin/env python3
"""
audit_season_coverage.py

Auditoría física y en tiempo real del clúster Sentinel-2 en MinIO y MongoDB.
Verifica la presencia real del raster (B02_10m.tif) en MinIO para los 487 tiles
canónicos del estudio a lo largo de las 4 estaciones.
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pymongo
from minio import Minio


def get_minio_client() -> Minio:
    host = os.getenv("MINIO_HOST", "192.168.219.61")
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
    host = os.getenv("MONGO_HOST", "192.168.219.61")
    port = os.getenv("MONGO_PORT", "31114")
    user = os.getenv("MONGO_USERNAME", "adminadmin")
    password = os.getenv("MONGO_PASSWORD", "adminadmin")
    db_name = os.getenv("MONGO_DB", "sentinel2-metadata")
    uri = f"mongodb://{user}:{password}@{host}:{port}/?authSource=admin"
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db_name]


def get_canonical_tiles() -> list:
    compose_path = Path("/home/diego/Virginia/landcoverpy/demo_deployment/docker-compose.yaml")
    if not compose_path.exists():
        compose_path = Path("/app/docker-compose.yaml")
    if not compose_path.exists():
        compose_path = Path("demo_deployment/docker-compose.yaml")

    if compose_path.exists():
        text = compose_path.read_text(encoding="utf-8")
        match = re.search(r"TILES_TO_PREDICT=(\[.*?\])", text)
        if match:
            return eval(match.group(1))

    # Fallback to env var if running in container
    env_tiles = os.getenv("TILES_TO_PREDICT")
    if env_tiles:
        if "#" in env_tiles:
            env_tiles = env_tiles.split("#")[0].strip()
        try:
            return json.loads(env_tiles)
        except Exception:
            pass

    return []


def main():
    print("=" * 86)
    print("      ESTADO REAL DEL CLÚSTER SENTINEL-2 (MEDITERRÁNEO 2021) - MINIO & MONGODB")
    print(f"      Fecha y hora: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 86)

    try:
        db = get_mongo_db()
        comp_col = db["composites"]
        minio_client = get_minio_client()
    except Exception as e:
        print(f"[ERROR] Conexión fallida a MongoDB/MinIO: {e}")
        sys.exit(1)

    tiles = get_canonical_tiles()
    if not tiles:
        print("[AVISO] No se pudo cargar TILES_TO_PREDICT desde docker-compose. Consultando Mongo...")
        tiles = sorted(comp_col.distinct("tile"))

    seasons = ["spring", "flowering", "summer", "autumn"]
    t0 = time.time()

    # Pre-cargar todos los metadatos de Mongo
    all_docs = list(comp_col.find({}, {"tile": 1, "season": 1, "periodLabel": 1, "S3BandsPrefix": 1}))

    tile_season_prefixes = defaultdict(list)
    for d in all_docs:
        s = d.get("season") or d.get("periodLabel")
        t = d.get("tile")
        p = d.get("S3BandsPrefix")
        if t and s and p:
            tile_season_prefixes[(t, s)].append(p)

    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")

    def check_tile_season(item):
        tile, season = item
        prefixes = tile_season_prefixes.get((tile, season), [])
        for p in prefixes:
            key = p.rstrip("/") + "/B02_10m.tif" if not p.endswith(".tif") else p
            try:
                minio_client.stat_object(bucket_composites, key)
                return (tile, season, True)
            except Exception:
                pass
        return (tile, season, False)

    all_items = [(t, s) for s in seasons for t in tiles]

    with ThreadPoolExecutor(max_workers=30) as pool:
        results = list(pool.map(check_tile_season, all_items))

    elapsed = time.time() - t0

    status_by_season = defaultdict(list)
    for t, s, is_ok in results:
        status_by_season[s].append((t, is_ok))

    print("\n1. PROGRESO FÍSICO POR ESTACIÓN (CAPAS VERIFICADAS EN MINIO):")
    print("-" * 86)
    print(f"{'Estación':<12} | {'Completados':<12} | {'% MinIO':<8} | {'Pendientes':<10} | {'Ejemplo / Lista Pendientes'}")
    print("-" * 86)

    total_ok = 0
    total_tiles = len(tiles)

    for s in seasons:
        ok_count = sum(1 for t, ok in status_by_season[s] if ok)
        missing = [t for t, ok in status_by_season[s] if not ok]
        pct = (ok_count / total_tiles) * 100 if total_tiles > 0 else 0
        total_ok += ok_count

        if len(missing) == 0:
            sample_str = "100% COMPLETADO (OK)"
        elif len(missing) <= 5:
            sample_str = f"Faltan: {missing}"
        else:
            sample_str = f"Faltan {len(missing)} (p. ej. {missing[:3]} ... {missing[-3:]})"

        print(f"{s.capitalize():<12} | {ok_count:>4} / {total_tiles:<5} | {pct:>6.1f}% | {len(missing):>10} | {sample_str}")

    total_expected = total_tiles * len(seasons)
    global_pct = (total_ok / total_expected) * 100 if total_expected > 0 else 0
    print("-" * 86)
    print(f"{'TOTAL GLOBAL':<12} | {total_ok:>4} / {total_expected:<5} | {global_pct:>6.1f}% | {total_expected - total_ok:>10} | Auditado en {elapsed:.2f}s")
    print("=" * 86)

    # 2. Última actividad registrada en el clúster
    print("\n2. ÚLTIMA ACTIVIDAD EN EL CLÚSTER (ÚLTIMOS 10 COMPOSITES GENERADOS):")
    print("-" * 86)
    recent = list(comp_col.find().sort("_id", -1).limit(10))
    for r in recent:
        t_gen = r["_id"].generation_time
        s_name = r.get("season") or r.get("periodLabel")
        tile_name = r.get("tile")
        title = r.get("title", "")
        # Cortar hash final si es largo
        print(f"  * {t_gen.strftime('%Y-%m-%d %H:%M:%S UTC')} | Estación: {s_name:<9} | Tile: {tile_name:<6} | {title}")

    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
