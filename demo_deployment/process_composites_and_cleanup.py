#!/usr/bin/env python3
"""
process_composites_and_cleanup.py

Herramienta de recuperacion, generacion de composites y purga de productos raw
para Sentinel-2 / LandCoverPy en MinIO y MongoDB.

Procesa los productos raw existentes en s2-products, genera los composites
estacionales en s2-composites y elimina los datos raw para liberar almacenamiento.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Limitar cache interna de GDAL para prevenir fugas de RAM
os.environ["GDAL_CACHEMAX"] = os.getenv("GDAL_CACHEMAX", "512")

import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configurar connection pooling para reutilizar sockets TCP en WSL2
_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    raise_on_status=False,
)
_GLOBAL_SESSION = requests.Session()
_ADAPTER = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=_RETRY_STRATEGY)
_GLOBAL_SESSION.mount("https://", _ADAPTER)
_GLOBAL_SESSION.mount("http://", _ADAPTER)
requests.get = _GLOBAL_SESSION.get

import pymongo
from minio import Minio

from landcoverpy.config import settings
from landcoverpy.composite import _create_composite, _validate_composite_products
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.minio import MinioConnection
from landcoverpy.mongo import MongoConnection
from landcoverpy.utilities.utils import get_products_by_tile_and_date


def get_minio_client() -> Minio:
    host = os.getenv("MINIO_HOST", "localhost")
    port = os.getenv("MINIO_PORT", "9000" if host != "localhost" else "31113")
    access_key = os.getenv("MINIO_ACCESS_KEY", "adminadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "adminadmin")
    return Minio(
        f"{host}:{port}",
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )


def get_mongo_db(max_retries: int = 20, delay: int = 3):
    host = os.getenv("MONGO_HOST", "localhost")
    port = os.getenv("MONGO_PORT", "27017" if host != "localhost" else "31114")
    user = os.getenv("MONGO_USERNAME", "adminadmin")
    password = os.getenv("MONGO_PASSWORD", "adminadmin")
    db_name = os.getenv("MONGO_DB", "sentinel2-metadata")
    uri = f"mongodb://{user}:{password}@{host}:{port}/"
    for attempt in range(1, max_retries + 1):
        try:
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            return client[db_name]
        except Exception as e:
            if attempt < max_retries:
                print(f"Esperando a que MongoDB termine de inicializarse ({attempt}/{max_retries})...")
                time.sleep(delay)
            else:
                raise e


def cleanup_tmp_dir():
    """Garantiza la limpieza de temporales locales."""
    tmp_dir = Path(settings.TMP_DIR)
    if tmp_dir.exists():
        for item in tmp_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass


def purge_raw_products_for_tile_season(
    minio_client: Minio,
    bucket_products: str,
    tile: str,
    products_metadata: List[dict],
    mongo_products_col=None,
) -> Tuple[int, int]:
    """
    Elimina los objetos raw de los productos especificados de s2-products en MinIO.
    Opcionalmente limpia los campos de indexes/intermediateProducts en MongoDB
    para liberar espacio en la base de datos manteniendo trazabilidad.
    Retorna (num_objetos_eliminados, bytes_liberados).
    """
    deleted_objects = 0
    deleted_bytes = 0

    for prod in products_metadata:
        prefix = prod.get("S3BandsPrefix")
        if prefix:
            if prefix.endswith("/raw/"):
                prefix = prefix[:-4]
            elif prefix.endswith("/raw"):
                prefix = prefix[:-3]
        else:
            # Construir el prefix desde el titulo del producto para no borrar otros tiles/estaciones
            title = prod.get("title", "")
            if "_T" in title:
                tile_part = title.split("_T")[1][:5]
                prefix = f"{tile_part}/{title.split('_')[2][:4]}/"
            else:
                print(f"  [ADVERTENCIA] Producto sin S3BandsPrefix ni titulo valido: {prod}. Saltando.")
                continue

        try:
            objects = list(minio_client.list_objects(bucket_products, prefix=prefix, recursive=True))
            for obj in objects:
                deleted_bytes += obj.size or 0
                minio_client.remove_object(bucket_products, obj.object_name)
                deleted_objects += 1
        except Exception as e:
            print(f"  [ADVERTENCIA] Error eliminando objetos de {prefix}: {e}")
            continue

        # Limpiar campos pesados en MongoDB manteniendo el documento del producto
        # (misma estrategia que sentinel2-download: mantener trazabilidad, liberar espacio)
        if mongo_products_col is not None:
            title = prod.get("title")
            if title:
                try:
                    mongo_products_col.update_many(
                        {"title": title},
                        {"$unset": {"indexes": "", "intermediateProducts": ""}}
                    )
                except Exception as e:
                    print(f"  [ADVERTENCIA] Error limpiando MongoDB para {title}: {e}")

    return deleted_objects, deleted_bytes


def verify_composite_exists(
    minio_client: Minio,
    mongo_composites_col,
    bucket_composites: str,
    tile: str,
    season: str,
    season_start: datetime,
    season_end: datetime,
) -> Optional[dict]:
    """
    Comprueba si el composite para (tile, season) ya existe tanto en MongoDB como en MinIO.
    Usa exclusivamente el indice compuesto (tile, season) para evitar falsos positivos
    con composites de otras estaciones.
    """
    comp_meta = mongo_composites_col.find_one({"tile": tile, "season": season})

    if comp_meta is not None:
        prefix = comp_meta.get("S3BandsPrefix")
        if prefix:
            try:
                objects = list(minio_client.list_objects(bucket_composites, prefix=prefix, recursive=True))
                # Un composite valido debe tener al menos 10 bandas
                if len(objects) >= 10:
                    return comp_meta
            except Exception:
                pass

    return None


def process_composites_and_cleanup(
    season_filter: Optional[str] = None,
    tile_filter: Optional[str] = None,
    dry_run: bool = False,
    num_workers: int = 1,
    worker_id: int = 0,
):
    print("=" * 80)
    print("LANDCOVERPY - PROCESAMIENTO DE COMPOSITES Y PURGA DE DATOS RAW")
    print(f"Modo: {'SIMULACION (DRY-RUN)' if dry_run else 'EJECUCION REAL'}")
    if num_workers > 1:
        print(f"Cluster distribuido: Worker {worker_id + 1} de {num_workers}")
    if season_filter:
        print(f"Filtro de estacion: {season_filter}")
    if tile_filter:
        print(f"Filtro de tile: {tile_filter}")
    print("=" * 80)

    minio_client = get_minio_client()
    mongo_db = get_mongo_db()
    mongo_products_col = mongo_db["products"]
    mongo_composites_col = mongo_db["composites"]

    # Asegurar indices compuestos en MongoDB
    mongo_composites_col.create_index([("tile", pymongo.ASCENDING), ("season", pymongo.ASCENDING)])
    mongo_composites_col.create_index([("title", pymongo.ASCENDING)], unique=True)
    mongo_products_col.create_index([("title", pymongo.ASCENDING)])

    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")

    seasons_file = os.getenv("SEASONS_FILE", "demo_deployment/app_data/seasons.json")
    if not Path(seasons_file).exists():
        seasons_file = "/app/data/seasons.json"

    with open(seasons_file, "r") as f:
        seasons = json.load(f)

    if season_filter and season_filter in seasons:
        seasons = {season_filter: seasons[season_filter]}

    # Detectar tiles con datos raw en s2-products o MongoDB
    print("\nDetectando tiles disponibles...")
    if tile_filter:
        target_tiles = [tile_filter]
    else:
        tiles_in_mongo = mongo_products_col.distinct("title")
        raw_tiles_set: Set[str] = set()
        for t in tiles_in_mongo:
            if "_T" in t:
                parts = t.split("_T")
                if len(parts) > 1 and len(parts[1]) >= 5:
                    raw_tiles_set.add(parts[1][:5])
        target_tiles = sorted(raw_tiles_set)

    total_target = len(target_tiles)
    print(f"Total tiles detectados con registros raw: {total_target}")

    if num_workers > 1:
        target_tiles = [t for i, t in enumerate(target_tiles) if i % num_workers == worker_id]
        print(f"Particion distribuida: Worker {worker_id + 1}/{num_workers} procesara {len(target_tiles)} tiles asignados.\n")
    else:
        print()

    min_useful_data_percentage = float(os.getenv("MIN_USEFUL_DATA_PERCENTAGE", "30"))
    max_products_composite = int(os.getenv("MAX_PRODUCTS_COMPOSITE", "5"))

    total_freed_bytes = 0
    total_composites_created = 0
    total_composites_skipped = 0

    for season_name, dates in seasons.items():
        season_start = datetime.strptime(dates["start"], "%Y-%m-%d")
        season_end = datetime.strptime(dates["end"], "%Y-%m-%d")

        print(f"\n>>> [ESTACION: {season_name.upper()}] ({dates['start']} a {dates['end']})")

        for idx, tile in enumerate(target_tiles, 1):
            t0 = time.time()
            prefix_info = f"[{idx}/{total_target}: {tile}] [{season_name}]"

            # 1. Comprobar si ya existe el composite en s2-composites
            existing_comp = verify_composite_exists(
                minio_client, mongo_composites_col, bucket_composites,
                tile, season_name, season_start, season_end
            )

            # Buscar productos raw en MongoDB
            cursor = get_products_by_tile_and_date(
                tile, mongo_products_col, season_start, season_end, min_useful_data_percentage
            )
            raw_products = list(cursor)

            if existing_comp is not None:
                total_composites_skipped += 1
                all_season_prods = list(mongo_products_col.find({
                    "tile": tile,
                    "datetakeSensingTime": {"$gte": season_start, "$lt": season_end}
                }))
                if all_season_prods:
                    if dry_run:
                        print(f"{prefix_info} Composite ya existe. [DRY-RUN] Se purgarian {len(all_season_prods)} productos raw residuales.")
                    else:
                        d_objs, d_bytes = purge_raw_products_for_tile_season(
                            minio_client, bucket_products, tile, all_season_prods,
                            mongo_products_col=mongo_products_col,
                        )
                        total_freed_bytes += d_bytes
                        print(f"{prefix_info} Composite ya existia. Purgados {d_objs} objetos raw residuales (+{d_bytes / (1024**3):.2f} GB).")
                else:
                    print(f"{prefix_info} Composite ya verificado. Omitiendo.")
                continue

            if not raw_products:
                print(f"{prefix_info} Sin productos raw validos (>= {min_useful_data_percentage}% utiles). Omitiendo.")
                continue

            # 2. Validar productos para el composite
            if dry_run:
                print(f"{prefix_info} [DRY-RUN] Se generaria composite con {len(raw_products[:max_products_composite])} productos y se purgarian los datos raw.")
                continue

            try:
                print(f"{prefix_info} Validando {len(raw_products)} capturas raw...")
                valid_products = _validate_composite_products(raw_products)
                selected_products = valid_products[:max_products_composite]

                if not selected_products:
                    print(f"{prefix_info} [ADVERTENCIA] Ningun producto supero la validacion de bandas. Omitiendo.")
                    continue

                print(f"{prefix_info} Generando composite con {len(selected_products)} productos...")
                _create_composite(
                    selected_products,
                    execution_mode=ExecutionMode.LAND_COVER_PREDICTION,
                    calculate_raw_indexes=True,
                    season=season_name,
                )

                # Verificar subida a MinIO
                comp_verified = verify_composite_exists(
                    minio_client, mongo_composites_col, bucket_composites,
                    tile, season_name, season_start, season_end
                )

                if comp_verified is None:
                    print(f"{prefix_info} [ERROR] La verificacion de integridad del composite fallo. No se borraran los raw.")
                    continue

                # Purgar datos raw de s2-products y limpiar campos en MongoDB (todas las capturas de la estacion)
                all_season_prods = list(mongo_products_col.find({
                    "tile": tile,
                    "datetakeSensingTime": {"$gte": season_start, "$lt": season_end}
                }))
                d_objs, d_bytes = purge_raw_products_for_tile_season(
                    minio_client, bucket_products, tile, all_season_prods or raw_products,
                    mongo_products_col=mongo_products_col,
                )
                total_freed_bytes += d_bytes
                total_composites_created += 1

                elapsed = time.time() - t0
                print(
                    f"{prefix_info} Composite completado y verificado en {elapsed:.1f}s | "
                    f"Purgados {d_objs} archivos raw (+{d_bytes / (1024**3):.2f} GB liberados)."
                )

            except Exception as e:
                print(f"{prefix_info} [ERROR] Excepcion al procesar composite: {e}")
            finally:
                cleanup_tmp_dir()
                gc.collect()

    print("\n" + "=" * 80)
    print("RESUMEN DE EJECUCION")
    print(f"Composites creados y verificados: {total_composites_created}")
    print(f"Composites ya existentes (omitidos): {total_composites_skipped}")
    print(f"Espacio total liberado en MinIO s2-products: {total_freed_bytes / (1024**3):.2f} GB ({total_freed_bytes / (1024**4):.2f} TB)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesamiento de Composites Sentinel-2 y Purga de Datos Raw"
    )
    parser.add_argument("--season", type=str, default=None, help="Filtrar por estacion (spring, flowering, summer, autumn)")
    parser.add_argument("--tile", type=str, default=None, help="Filtrar por tile especifico (ej. 30STH)")
    parser.add_argument("--dry-run", action="store_true", help="Simulacion sin realizar cambios ni borrados")
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "1")), help="Numero total de maquinas/workers en paralelo")
    parser.add_argument("--worker-id", type=int, default=int(os.getenv("WORKER_ID", "0")), help="ID de este worker (0 a num-workers - 1)")

    args = parser.parse_args()
    process_composites_and_cleanup(
        season_filter=args.season,
        tile_filter=args.tile,
        dry_run=args.dry_run,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
    )

