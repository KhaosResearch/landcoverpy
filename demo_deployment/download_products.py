import contextlib
import gc
import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# Limit GDAL internal cache to prevent RAM exhaustion
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

# Configure Connection Pooling for TCP socket reuse in WSL2/Linux
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

# Replace requests.get globally so all ds_download calls reuse connections
requests.get = _GLOBAL_SESSION.get


def _reset_global_session():
    """Reset the global HTTP session to clear stale sockets."""
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
    """Context manager to suppress low-level stdout spam (e.g. JP2 individual upload spam)."""
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
                print(f"Waiting for MongoDB initialization ({attempt}/{max_retries})...")
                time.sleep(delay)
            else:
                raise e


def _cleanup_tmp_dir():
    """Ensure residual temporary files in TMP_DIR are deleted."""
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
    """Check if composite already exists and contains its layers in s2-composites."""
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
    Deletes ALL raw files, intermediate products, and cloudy scenes from s2-products
    in MinIO for the tile across all months of the corresponding season.
    Uses batch deletion (DeleteObject) for maximum throughput and complete cleanup.
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
                            print(f"    [WARNING] Error deleting object {err.name}: {err.message}")
                deleted_objects += len(raw_objs)
        except Exception as e:
            print(f"    [WARNING] Error scanning/deleting objects from {month_prefix}: {e}")

    # Clean heavy metadata fields in MongoDB for all captures of that tile in the period
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
            print(f"    [WARNING] Error cleaning MongoDB metadata for {tile} ({season}): {e}")

    return deleted_objects, deleted_bytes


def _acquire_tile_claim(
    mongo_db,
    tile: str,
    season: str,
    worker_id: str,
    timeout_seconds: int = 7200,
) -> bool:
    """
    Atomically acquire a claim on a tile to prevent concurrent workers from
    processing the same tile simultaneously (especially during mid-pipeline crossover).
    Returns True if tile was claimed successfully, False if already claimed by another worker.
    """
    col = mongo_db["tile_claims"]
    now = datetime.utcnow()
    expire_threshold = now - timedelta(seconds=timeout_seconds)

    try:
        # 1. If an expired claim (> 2h) or our own claim exists, refresh it
        res = col.find_one_and_update(
            {
                "tile": tile,
                "season": season,
                "$or": [
                    {"claimed_at": {"$lt": expire_threshold}},
                    {"claimed_by": worker_id},
                ],
            },
            {
                "$set": {
                    "claimed_by": worker_id,
                    "claimed_at": now,
                    "status": "processing",
                }
            },
            upsert=False,
        )
        if res is not None:
            return True

        # 2. If no document existed, attempt atomic creation via upsert
        col.update_one(
            {"tile": tile, "season": season},
            {
                "$setOnInsert": {
                    "tile": tile,
                    "season": season,
                    "claimed_by": worker_id,
                    "claimed_at": now,
                    "status": "processing",
                }
            },
            upsert=True,
        )
        doc = col.find_one({"tile": tile, "season": season})
        return doc is not None and doc.get("claimed_by") == worker_id
    except Exception:
        # Unique index collision upon concurrent upsert
        return False


def _release_tile_claim(
    mongo_db,
    tile: str,
    season: str,
    worker_id: str,
    success: bool = True,
):
    """Release or update the tile claim state."""
    try:
        col = mongo_db["tile_claims"]
        if success:
            col.update_one(
                {"tile": tile, "season": season, "claimed_by": worker_id},
                {"$set": {"status": "completed", "completed_at": datetime.utcnow()}},
            )
        else:
            col.delete_one({"tile": tile, "season": season, "claimed_by": worker_id})
    except Exception:
        pass


def download_products():
    seasons_file = os.getenv("SEASONS_FILE", "/app/data/seasons.json")
    if not Path(seasons_file).exists():
        seasons_file = "demo_deployment/app_data/seasons.json"

    with open(seasons_file, "r") as f:
        seasons = json.load(f)

    target_season = os.getenv("TARGET_SEASON")
    if target_season:
        requested = [s.strip().lower() for s in target_season.split(",") if s.strip()]
        valid_seasons = {k: v for k, v in seasons.items() if k.lower() in requested}
        if valid_seasons:
            seasons = valid_seasons
            print(f"Filtering execution for seasons: {list(seasons.keys())}")
        else:
            print(
                f"[WARNING] TARGET_SEASON='{target_season}' is invalid. Options: {list(seasons.keys())}. Processing all."
            )

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

    reverse_tiles = os.getenv("REVERSE_TILES", "false").lower() in ("true", "1", "yes")
    tiles_to_download = sorted(tiles_to_train.union(tiles_to_predict), reverse=reverse_tiles)
    total_tiles = len(tiles_to_download)
    total_seasons = len(seasons)
    total_expected_composites = total_tiles * total_seasons

    worker_id = os.getenv("WORKER_ID", os.getenv("HOSTNAME", f"worker-{os.getpid()}"))

    minio_client = _get_minio_client()
    mongo_db = _get_mongo_db()
    mongo_products_col = mongo_db["products"]
    mongo_composites_col = mongo_db["composites"]
    mongo_claims_col = mongo_db["tile_claims"]

    # Ensure compound indexes in MongoDB for O(1) checks and concurrency control
    mongo_composites_col.create_index([("tile", pymongo.ASCENDING), ("season", pymongo.ASCENDING)])
    mongo_composites_col.create_index([("title", pymongo.ASCENDING)], unique=True)
    mongo_products_col.create_index([("title", pymongo.ASCENDING)])
    mongo_claims_col.create_index([("tile", pymongo.ASCENDING), ("season", pymongo.ASCENDING)], unique=True)

    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")

    min_useful_data_percentage = float(os.getenv("MIN_USEFUL_DATA_PERCENTAGE", "30"))
    max_products_composite = int(os.getenv("MAX_PRODUCTS_COMPOSITE", "5"))

    print("=" * 80)
    print("LANDCOVERPY - SENTINEL-2 COMPOSITE PIPELINE (MEDITERRANEAN 2021)")
    print(f"Total Tiles: {total_tiles} | Seasons: {total_seasons} | Total Composites: {total_expected_composites}")
    print(f"Worker ID: {worker_id} | Direction: {'REVERSE (Z -> A)' if reverse_tiles else 'STANDARD (A -> Z)'}")
    print("Strategy: Season-by-Season | Retention: Composites Only (Immediate Raw Purge)")
    print("=" * 80)

    cumulative_completed = 0
    failed_tiles_dict = {}

    for season_idx, (season_name, dates) in enumerate(seasons.items(), 1):
        start_date = datetime.strptime(dates["start"], "%Y-%m-%d")
        end_date = datetime.strptime(dates["end"], "%Y-%m-%d")

        print(f"\n" + "-" * 80)
        print(f">>> [SEASON {season_idx}/{total_seasons}: {season_name.upper()}] ({dates['start']} to {dates['end']})")
        print("-" * 80)

        # Initial count of existing composites for this season
        existing_in_season = mongo_composites_col.count_documents({"season": season_name})
        print(f"Initial status: {existing_in_season}/{total_tiles} tiles completed ({existing_in_season / total_tiles * 100:.1f}%) | {total_tiles - existing_in_season} pending.")

        failed_tiles_dict[season_name] = []

        for tile_idx, tile in enumerate(tiles_to_download, 1):
            t_start = time.time()
            prefix_log = f"[{tile_idx}/{total_tiles}: {tile}] [{season_name}]"

            # 1. Check if composite already exists in s2-composites
            comp_existing = _verify_composite_in_minio(
                minio_client, mongo_composites_col, bucket_composites,
                tile, season_name, start_date, end_date
            )

            if comp_existing is not None:
                # If composite exists, check for residual raw files and purge them
                d_objs, d_bytes = _purge_raw_products(
                    minio_client, bucket_products, tile, season_name,
                    start_date, end_date, mongo_products_col=mongo_products_col,
                )
                if d_objs > 0:
                    print(f"  {prefix_log} Existing composite verified. Purged {d_objs} residual raw objects (+{d_bytes / (1024**3):.2f} GB).")
                else:
                    print(f"  {prefix_log} Already processed (OK).")

                cumulative_completed += 1
                continue

            # 2. Check atomic claim to prevent concurrent worker collisions (crossover)
            if not _acquire_tile_claim(mongo_db, tile, season_name, worker_id):
                claim_doc = mongo_claims_col.find_one({"tile": tile, "season": season_name})
                claimed_by_info = claim_doc.get("claimed_by", "another worker") if claim_doc else "another worker"
                print(f"  {prefix_log} [ACTIVE CLAIM] Tile reserved by {claimed_by_info}. Skipping to avoid duplication.")
                continue

            # 3. If composite does not exist: download/process with retries and cooldown for WSL2
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
                        print(f"  {prefix_log} Downloading captures from Google Cloud Sentinel API (attempt {attempt}/{max_tile_attempts})...")
                        # Silence low-level individual JP2 upload spam
                        with SuppressStdout():
                            download_product_using_sentinel_api(
                                False, True, start_date, end_date, tile_id=tile
                            )

                        # Re-query MongoDB after download
                        cursor = get_products_by_tile_and_date(
                            tile, mongo_products_col, start_date, end_date, min_useful_data_percentage
                        )
                        raw_products = list(cursor)

                    if not raw_products:
                        print(f"  {prefix_log} [WARNING] No captures found with >= {min_useful_data_percentage}% useful data.")
                        failed_tiles_dict[season_name].append(tile)
                        break

                    # 4. Validate and select best acquisitions
                    print(f"  {prefix_log} Validating {len(raw_products)} captures for composite...")
                    valid_products = _validate_composite_products(raw_products)
                    selected_products = valid_products[:max_products_composite]

                    if not selected_products:
                        print(f"  {prefix_log} [WARNING] No captures passed band validation.")
                        failed_tiles_dict[season_name].append(tile)
                        break

                    # 5. Generate composite (pixel-by-pixel median and spectral indices)
                    print(f"  {prefix_log} Calculating median and indices over {len(selected_products)} captures...")
                    _create_composite(
                        selected_products,
                        execution_mode=ExecutionMode.LAND_COVER_PREDICTION,
                        calculate_raw_indexes=True,
                        season=season_name,
                    )

                    # 6. Verify composite was successfully uploaded to s2-composites
                    comp_verified = _verify_composite_in_minio(
                        minio_client, mongo_composites_col, bucket_composites,
                        tile, season_name, start_date, end_date
                    )

                    if comp_verified is None:
                        print(f"  {prefix_log} [ERROR] Composite verification in s2-composites failed. Raw files will not be deleted.")
                        failed_tiles_dict[season_name].append(tile)
                        break

                    # 7. Immediately purge raw products from s2-products and clean MongoDB
                    d_objs, d_bytes = _purge_raw_products(
                        minio_client, bucket_products, tile, season_name,
                        start_date, end_date, mongo_products_col=mongo_products_col,
                    )

                    elapsed = time.time() - t_start
                    cumulative_completed += 1
                    season_pct = (tile_idx / total_tiles) * 100
                    total_pct = (cumulative_completed / total_expected_composites) * 100

                    print(
                        f"  {prefix_log} Composite saved in {elapsed:.1f}s | "
                        f"Purged {d_objs} raw (+{d_bytes / (1024**3):.2f} GB) | "
                        f"Season: {tile_idx}/{total_tiles} ({season_pct:.1f}%) | "
                        f"Total: {cumulative_completed}/{total_expected_composites} ({total_pct:.1f}%)"
                    )
                    tile_success = True

                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, urllib3.exceptions.HTTPError) as net_err:
                    print(f"  {prefix_log} [WARNING] Network error / socket saturation detected: {net_err}")
                    if attempt < max_tile_attempts:
                        cooldown_seconds = 180  # 3-minute cooldown pause to allow kernel to release TIME_WAIT sockets
                        print(f"  {prefix_log} [COOLDOWN PAUSE] Pausing for {cooldown_seconds}s to allow kernel to release sockets...")
                        time.sleep(cooldown_seconds)
                        _reset_global_session()
                    else:
                        print(f"  {prefix_log} [ERROR] Exhausted {max_tile_attempts} network retries on tile {tile}.")
                        failed_tiles_dict[season_name].append(tile)
                except Exception as e:
                    print(f"  {prefix_log} [ERROR] Failed to process tile: {e}")
                    failed_tiles_dict[season_name].append(tile)
                    break
                finally:
                    _cleanup_tmp_dir()
                    gc.collect()

            # Release or update claim based on success/failure
            _release_tile_claim(mongo_db, tile, season_name, worker_id, success=tile_success)

        # Season summary
        fails = len(failed_tiles_dict[season_name])
        print(f"\n>>> [END OF SEASON {season_name.upper()}] Successfully completed: {total_tiles - fails}/{total_tiles} | Failed/Pending: {fails}")

    print("\n" + "=" * 80)
    print("COMPOSITE GENERATION PIPELINE FINISHED")
    print(f"Composites completed in s2-composites: {cumulative_completed} / {total_expected_composites}")
    print("=" * 80)


if __name__ == "__main__":
    download_products()
