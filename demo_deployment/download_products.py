import contextlib
import gc
import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Limitar cache interna de GDAL para prevenir fugas de RAM
os.environ["GDAL_CACHEMAX"] = os.getenv("GDAL_CACHEMAX", "512")

import pymongo
from minio import Minio

from ds_download.download_using_sentinel_api import download_product_using_sentinel_api
from landcoverpy.config import settings
from landcoverpy.composite import _create_composite, _validate_composite_products
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.utilities.geometries import (
    _csv_to_geojson,
    _group_validated_data_points_by_tile,
    _kmz_to_geojson,
)
from landcoverpy.utilities.utils import get_products_by_tile_and_date

import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

# Configuración de Connection Pooling para reutilización de sockets TCP en WSL2
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

# Reemplazar requests.get a nivel global para que todas las llamadas de ds_download reutilicen conexiones
requests.get = _GLOBAL_SESSION.get


def _reset_global_session():
    """Reinicia la sesión HTTP global para limpiar conectores estancados."""
    global _GLOBAL_SESSION
    try:
        _GLOBAL_SESSION.close()
    except Exception:
        pass
    _GLOBAL_SESSION = requests.Session()
    adapter = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=_RETRY_STRATEGY)
    _GLOBAL_SESSION.mount("https://", adapter)
    _GLOBAL_SESSION.mount("http://", adapter)
    requests.get = _GLOBAL_SESSION.get


class SuppressStdout:
    """Context manager para silenciar salidas de bajo nivel (spam de subida JP2)."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout


def _get_minio_client() -> Minio:
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


def _get_mongo_db(max_retries: int = 20, delay: int = 3):
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


def _cleanup_tmp_dir():
    """Garantiza la limpieza de archivos temporales residuales en TMP_DIR."""
    tmp_dir = Path(os.getenv("TMP_DIR", "/app/tmp/"))
    if tmp_dir.exists():
        for item in tmp_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass


def _verify_composite_in_minio(
    minio_client: Minio,
    mongo_composites_col,
    bucket_composites: str,
    tile: str,
    season: str,
    season_start: datetime,
    season_end: datetime,
) -> Optional[dict]:
    """Comprueba si el composite ya existe y tiene sus capas en s2-composites."""
    comp_meta = mongo_composites_col.find_one({"tile": tile, "season": season})

    if comp_meta is not None:
        prefix = comp_meta.get("S3BandsPrefix")
        if prefix:
            try:
                objects = list(minio_client.list_objects(bucket_composites, prefix=prefix, recursive=True))
                if len(objects) >= 10:
                    return comp_meta
            except Exception:
                pass

    return None


SEASON_MONTHS = {
    "spring": ["March", "April"],
    "flowering": ["May"],
    "summer": ["June", "July"],
    "autumn": ["October", "November"],
}


def _purge_raw_products(
    minio_client: Minio,
    bucket_products: str,
    tile: str,
    season: str,
    start_date: datetime,
    end_date: datetime,
    mongo_products_col=None,
) -> Tuple[int, int]:
    """
    Elimina TODOS los archivos raw, intermedios y escenas nubladas de s2-products
    en MinIO para el tile y todos los meses de la estacion correspondiente.
    Utiliza borrado por lotes (DeleteObject) para maxima velocidad y limpieza total.
    """
    from minio.deleteobjects import DeleteObject

    deleted_objects = 0
    deleted_bytes = 0
    months = SEASON_MONTHS.get(season, [])

    for month in months:
        month_prefix = f"{tile}/2021/{month}/"
        try:
            raw_objs = list(minio_client.list_objects(bucket_products, prefix=month_prefix, recursive=True))
            if raw_objs:
                for obj in raw_objs:
                    deleted_bytes += obj.size or 0
                del_list = [DeleteObject(o.object_name) for o in raw_objs]
                for c_i in range(0, len(del_list), 1000):
                    del_chunk = del_list[c_i:c_i + 1000]
                    errors = list(minio_client.remove_objects(bucket_products, del_chunk))
                    if errors:
                        for err in errors:
                            print(f"    [ADVERTENCIA] Error eliminando objeto {err.name}: {err.message}")
                deleted_objects += len(raw_objs)
        except Exception as e:
            print(f"    [ADVERTENCIA] Error escaneando/eliminando objetos de {month_prefix}: {e}")

    # Limpiar campos pesados en MongoDB para todas las capturas de ese tile en el periodo
    if mongo_products_col is not None:
        try:
            mongo_products_col.update_many(
                {
                    "tile": tile,
                    "datetakeSensingTime": {"$gte": start_date, "$lt": end_date},
                },
                {"$unset": {"indexes": "", "intermediateProducts": ""}}
            )
        except Exception as e:
            print(f"    [ADVERTENCIA] Error limpiando MongoDB para {tile} ({season}): {e}")

    return deleted_objects, deleted_bytes


def download_products():
    seasons_file = os.getenv("SEASONS_FILE", "/app/data/seasons.json")
    if not Path(seasons_file).exists():
        seasons_file = "demo_deployment/app_data/seasons.json"

    with open(seasons_file, "r") as f:
        seasons = json.load(f)

    target_season = os.getenv("TARGET_SEASON")
    if target_season:
        target_season = target_season.lower().strip()
        if target_season in seasons:
            seasons = {target_season: seasons[target_season]}
            print(f"Filtrando ejecucion para estacion unica: {target_season.upper()}")
        else:
            print(f"[ADVERTENCIA] TARGET_SEASON='{target_season}' no valida. Opciones: {list(seasons.keys())}. Procesando todas.")

    data_file = os.getenv("DB_FILE")
    if data_file:
        if data_file.endswith(".kmz"):
            data_file = _kmz_to_geojson(data_file)
        if data_file.endswith(".csv"):
            data_file = _csv_to_geojson(data_file, sep=";")
        polygons_per_tile = _group_validated_data_points_by_tile(data_file)
        tiles_to_train = set(polygons_per_tile.keys())
    else:
        tiles_to_train = set()

    tiles_to_predict_env = os.getenv("TILES_TO_PREDICT", "[]")
    if "#" in tiles_to_predict_env:
        tiles_to_predict_env = tiles_to_predict_env.split("#")[0].strip()
    if tiles_to_predict_env.lower() != "prediction":
        try:
            tiles_to_predict = set(json.loads(tiles_to_predict_env))
        except json.JSONDecodeError:
            raise ValueError(
                f"Error parsing TILES_TO_PREDICT: {tiles_to_predict_env}."
            )
    else:
        tiles_to_predict = set()

    tiles_to_download = sorted(tiles_to_train.union(tiles_to_predict))
    total_tiles = len(tiles_to_download)
    total_seasons = len(seasons)
    total_expected_composites = total_tiles * total_seasons

    minio_client = _get_minio_client()
    mongo_db = _get_mongo_db()
    mongo_products_col = mongo_db["products"]
    mongo_composites_col = mongo_db["composites"]

    # Asegurar indices compuestos en MongoDB para comprobaciones O(1)
    mongo_composites_col.create_index([("tile", pymongo.ASCENDING), ("season", pymongo.ASCENDING)])
    mongo_composites_col.create_index([("title", pymongo.ASCENDING)], unique=True)
    mongo_products_col.create_index([("title", pymongo.ASCENDING)])

    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")

    min_useful_data_percentage = float(os.getenv("MIN_USEFUL_DATA_PERCENTAGE", "30"))
    max_products_composite = int(os.getenv("MAX_PRODUCTS_COMPOSITE", "5"))

    print("=" * 80)
    print("LANDCOVERPY - PIPELINE DE COMPOSITES SENTINEL-2 (MEDITERRANEO 2021)")
    print(f"Total Tiles: {total_tiles} | Estaciones: {total_seasons} | Total Composites: {total_expected_composites}")
    print("Estrategia: Season-by-Season | Retencion: Composites Solo (Purga Inmediata de Raw)")
    print("=" * 80)

    cumulative_completed = 0
    failed_tiles_dict = {}

    for season_idx, (season_name, dates) in enumerate(seasons.items(), 1):
        start_date = datetime.strptime(dates["start"], "%Y-%m-%d")
        end_date = datetime.strptime(dates["end"], "%Y-%m-%d")

        print(f"\n" + "-" * 80)
        print(f">>> [ESTACION {season_idx}/{total_seasons}: {season_name.upper()}] ({dates['start']} a {dates['end']})")
        print("-" * 80)

        # Conteo inicial de composites ya existentes para esta estacion
        existing_in_season = mongo_composites_col.count_documents({"season": season_name})
        print(f"Estado inicial: {existing_in_season}/{total_tiles} tiles completados ({existing_in_season / total_tiles * 100:.1f}%) | {total_tiles - existing_in_season} pendientes.")

        failed_tiles_dict[season_name] = []

        for tile_idx, tile in enumerate(tiles_to_download, 1):
            t_start = time.time()
            prefix_log = f"[{tile_idx}/{total_tiles}: {tile}] [{season_name}]"

            # 1. Comprobar si el composite ya existe en s2-composites
            comp_existing = _verify_composite_in_minio(
                minio_client, mongo_composites_col, bucket_composites,
                tile, season_name, start_date, end_date
            )

            if comp_existing is not None:
                # Si existe el composite, verificar si quedan datos raw residuales y purgarlos
                d_objs, d_bytes = _purge_raw_products(
                    minio_client, bucket_products, tile, season_name,
                    start_date, end_date, mongo_products_col=mongo_products_col,
                )
                if d_objs > 0:
                    print(f"  {prefix_log} Composite existente verificado. Purgados {d_objs} raw residuales (+{d_bytes / (1024**3):.2f} GB).")
                else:
                    print(f"  {prefix_log} Ya procesado (OK).")

                cumulative_completed += 1
                continue

            # 2. Si no existe composite: descargar/procesar con reintentos y enfriamiento para WSL2
            max_tile_attempts = 3
            attempt = 0
            tile_success = False

            while attempt < max_tile_attempts and not tile_success:
                attempt += 1
                try:
                    cursor = get_products_by_tile_and_date(
                        tile, mongo_products_col, start_date, end_date, min_useful_data_percentage
                    )
                    raw_products = list(cursor)

                    if not raw_products:
                        print(f"  {prefix_log} Descargando capturas desde Google Cloud Sentinel API (intento {attempt}/{max_tile_attempts})...")
                        # Silenciar spam de subida individual JP2
                        with SuppressStdout():
                            download_product_using_sentinel_api(
                                False, True, start_date, end_date, tile_id=tile
                            )

                        # Reconsultar MongoDB tras la descarga
                        cursor = get_products_by_tile_and_date(
                            tile, mongo_products_col, start_date, end_date, min_useful_data_percentage
                        )
                        raw_products = list(cursor)

                    if not raw_products:
                        print(f"  {prefix_log} [ADVERTENCIA] No se encontraron capturas con >= {min_useful_data_percentage}% datos utiles.")
                        failed_tiles_dict[season_name].append(tile)
                        break

                    # 3. Validar y seleccionar las mejores adquisiciones
                    print(f"  {prefix_log} Validando {len(raw_products)} capturas para composite...")
                    valid_products = _validate_composite_products(raw_products)
                    selected_products = valid_products[:max_products_composite]

                    if not selected_products:
                        print(f"  {prefix_log} [ADVERTENCIA] Ninguna captura supero la validacion de bandas.")
                        failed_tiles_dict[season_name].append(tile)
                        break

                    # 4. Generar composite (mediana pixel a pixel e indices espectrales)
                    print(f"  {prefix_log} Calculando mediana e indices sobre {len(selected_products)} capturas...")
                    _create_composite(
                        selected_products,
                        execution_mode=ExecutionMode.LAND_COVER_PREDICTION,
                        calculate_raw_indexes=True,
                        season=season_name,
                    )

                    # 5. Verificar que el composite se subio correctamente a s2-composites
                    comp_verified = _verify_composite_in_minio(
                        minio_client, mongo_composites_col, bucket_composites,
                        tile, season_name, start_date, end_date
                    )

                    if comp_verified is None:
                        print(f"  {prefix_log} [ERROR] La verificacion del composite en s2-composites fallo. No se eliminaran los raw.")
                        failed_tiles_dict[season_name].append(tile)
                        break

                    # 6. Purgar datos raw de s2-products inmediatamente y limpiar MongoDB
                    d_objs, d_bytes = _purge_raw_products(
                        minio_client, bucket_products, tile, season_name,
                        start_date, end_date, mongo_products_col=mongo_products_col,
                    )

                    elapsed = time.time() - t_start
                    cumulative_completed += 1
                    season_pct = (tile_idx / total_tiles) * 100
                    total_pct = (cumulative_completed / total_expected_composites) * 100

                    print(
                        f"  {prefix_log} Composite guardado en {elapsed:.1f}s | "
                        f"Purgados {d_objs} raw (+{d_bytes / (1024**3):.2f} GB) | "
                        f"Estacion: {tile_idx}/{total_tiles} ({season_pct:.1f}%) | "
                        f"Total: {cumulative_completed}/{total_expected_composites} ({total_pct:.1f}%)"
                    )
                    tile_success = True

                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, urllib3.exceptions.HTTPError) as net_err:
                    print(f"  {prefix_log} [ADVERTENCIA] Error de red / saturacion de sockets WSL2 detectado: {net_err}")
                    if attempt < max_tile_attempts:
                        cooldown_seconds = 180  # Pausa de enfriamiento de 3 minutos para que WSL2 libere sockets TIME_WAIT
                        print(f"  {prefix_log} [PAUSA DE ENFRIAMIENTO] Pausando durante {cooldown_seconds}s para permitir al kernel liberar conectores...")
                        time.sleep(cooldown_seconds)
                        _reset_global_session()
                    else:
                        print(f"  {prefix_log} [ERROR] Agotados {max_tile_attempts} reintentos de red en tile {tile}.")
                        failed_tiles_dict[season_name].append(tile)
                except Exception as e:
                    print(f"  {prefix_log} [ERROR] Fallo al procesar tile: {e}")
                    failed_tiles_dict[season_name].append(tile)
                    break
                finally:
                    _cleanup_tmp_dir()
                    gc.collect()

        # Resumen de estacion
        fails = len(failed_tiles_dict[season_name])
        print(f"\n>>> [FIN ESTACION {season_name.upper()}] Completados con exito: {total_tiles - fails}/{total_tiles} | Fallidos/Pendientes: {fails}")

    print("\n" + "=" * 80)
    print("PROCESO DE GENERACION DE COMPOSITES FINALIZADO")
    print(f"Composites completados en s2-composites: {cumulative_completed} / {total_expected_composites}")
    print("=" * 80)


if __name__ == "__main__":
    download_products()
