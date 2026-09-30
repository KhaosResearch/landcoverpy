#!/usr/bin/env python3
"""
process_composites_and_cleanup.py

Recovery, composite generation, and raw products purge utility
for Sentinel-2 / LandCoverPy in MinIO and MongoDB.

Processes existing raw products in s2-products, generates seasonal
composites in s2-composites, and purges raw captures to immediately reclaim storage.
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

# Limit GDAL internal cache to prevent RAM exhaustion
os.environ["GDAL_CACHEMAX"] = os.getenv("GDAL_CACHEMAX", "512")

import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure connection pooling to reuse TCP sockets in WSL2/Linux
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
                print(f"Waiting for MongoDB initialization ({attempt}/{max_retries})...")
                time.sleep(delay)
            else:
                raise e


def cleanup_tmp_dir():
    """Ensure local temporary files are completely purged."""
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


SEASON_MONTHS = {
    "spring": ["March", "April"],
    "flowering": ["May"],
    "summer": ["June", "July"],
    "autumn": ["October", "November"],
}


def purge_raw_products_for_tile_season(
    minio_client: Minio,
    bucket_products: str,
    tile: str,
    season: str,
    season_start: datetime,
    season_end: datetime,
    mongo_products_col=None,
) -> Tuple[int, int]:
    """
    Deletes ALL raw objects, intermediate products, and cloudy captures in s2-products
    for the specified tile across all months of the given season.
    Uses batch deletion (DeleteObject) for maximum throughput and complete cleanup.
    Returns (num_deleted_objects, freed_bytes).
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
                            print(f"  [ERROR] Deleting {err.name}: {err.message}")
                deleted_objects += len(raw_objs)
        except Exception as e:
            print(f"  [WARNING] Error deleting objects from {month_prefix}: {e}")

    # Clean heavy metadata fields in MongoDB while keeping the product document
    if mongo_products_col is not None:
        try:
            mongo_products_col.update_many(
                {
                    "tile": tile,
                    "datetakeSensingTime": {"$gte": season_start, "$lt": season_end},
                },
                {"$unset": {"indexes": "", "intermediateProducts": ""}}
            )
        except Exception as e:
            print(f"  [WARNING] Error cleaning MongoDB metadata for {tile} ({season}): {e}")

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
    Checks whether a composite for (tile, season) already exists in both MongoDB and MinIO.
    Exclusively checks the compound index (tile, season) to avoid false positives
    with composites from different seasons.
    """
    comp_meta = mongo_composites_col.find_one({"tile": tile, "season": season})

    if comp_meta is not None:
        prefix = comp_meta.get("S3BandsPrefix")
        if prefix:
            try:
                objects = list(minio_client.list_objects(bucket_composites, prefix=prefix, recursive=True))
                # A complete composite must contain at least 10 bands
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
    print("LANDCOVERPY - COMPOSITE PROCESSING AND RAW DATA PURGE")
    print(f"Mode: {'SIMULATION (DRY-RUN)' if dry_run else 'LIVE EXECUTION'}")
    if num_workers > 1:
        print(f"Distributed cluster: Worker {worker_id + 1} of {num_workers}")
    if season_filter:
        print(f"Season filter: {season_filter}")
    if tile_filter:
        print(f"Tile filter: {tile_filter}")
    print("=" * 80)

    minio_client = get_minio_client()
    mongo_db = get_mongo_db()
    mongo_products_col = mongo_db["products"]
    mongo_composites_col = mongo_db["composites"]

    # Ensure compound indexes in MongoDB
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

    # Detect tiles with raw data in s2-products or MongoDB
    print("\nDetecting available tiles...")
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
    print(f"Total tiles detected with raw records: {total_target}")

    if num_workers > 1:
        target_tiles = [t for i, t in enumerate(target_tiles) if i % num_workers == worker_id]
        print(f"Distributed partition: Worker {worker_id + 1}/{num_workers} processing {len(target_tiles)} assigned tiles.\n")
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

        print(f"\n>>> [SEASON: {season_name.upper()}] ({dates['start']} to {dates['end']})")

        for idx, tile in enumerate(target_tiles, 1):
            t0 = time.time()
            prefix_info = f"[{idx}/{total_target}: {tile}] [{season_name}]"

            # 1. Check if composite already exists in s2-composites
            existing_comp = verify_composite_exists(
                minio_client, mongo_composites_col, bucket_composites,
                tile, season_name, season_start, season_end
            )

            # Query raw products in MongoDB
            cursor = get_products_by_tile_and_date(
                tile, mongo_products_col, season_start, season_end, min_useful_data_percentage
            )
            raw_products = list(cursor)

            if existing_comp is not None:
                total_composites_skipped += 1
                if dry_run:
                    print(f"{prefix_info} Composite already exists. [DRY-RUN] Would verify residual raw files.")
                else:
                    d_objs, d_bytes = purge_raw_products_for_tile_season(
                        minio_client, bucket_products, tile, season_name,
                        season_start, season_end,
                        mongo_products_col=mongo_products_col,
                    )
                    if d_objs > 0:
                        total_freed_bytes += d_bytes
                        print(f"{prefix_info} Existing composite verified. Purged {d_objs} residual raw objects (+{d_bytes / (1024**3):.2f} GB).")
                    else:
                        print(f"{prefix_info} Composite already verified. Skipping.")
                continue

            if not raw_products:
                print(f"{prefix_info} No valid raw products (>= {min_useful_data_percentage}% useful). Skipping.")
                continue

            # 2. Validate products for composite
            if dry_run:
                print(f"{prefix_info} [DRY-RUN] Would generate composite with {len(raw_products[:max_products_composite])} products and purge raw data.")
                continue

            try:
                print(f"{prefix_info} Validating {len(raw_products)} raw captures...")
                valid_products = _validate_composite_products(raw_products)
                selected_products = valid_products[:max_products_composite]

                if not selected_products:
                    print(f"{prefix_info} [WARNING] No products passed band validation. Skipping.")
                    continue

                print(f"{prefix_info} Generating composite with {len(selected_products)} products...")
                _create_composite(
                    selected_products,
                    execution_mode=ExecutionMode.LAND_COVER_PREDICTION,
                    calculate_raw_indexes=True,
                    season=season_name,
                )

                # Verify upload in MinIO
                comp_verified = verify_composite_exists(
                    minio_client, mongo_composites_col, bucket_composites,
                    tile, season_name, season_start, season_end
                )

                if comp_verified is None:
                    print(f"{prefix_info} [ERROR] Composite integrity check failed in MinIO. Raw files will not be deleted.")
                    continue

                # Purge raw products from s2-products and clean MongoDB metadata
                d_objs, d_bytes = purge_raw_products_for_tile_season(
                    minio_client, bucket_products, tile, season_name,
                    season_start, season_end,
                    mongo_products_col=mongo_products_col,
                )
                total_freed_bytes += d_bytes
                total_composites_created += 1

                elapsed = time.time() - t0
                print(
                    f"{prefix_info} Composite completed and verified in {elapsed:.1f}s | "
                    f"Purged {d_objs} raw files (+{d_bytes / (1024**3):.2f} GB freed)."
                )

            except Exception as e:
                print(f"{prefix_info} [ERROR] Exception processing composite: {e}")
            finally:
                cleanup_tmp_dir()
                gc.collect()

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print(f"Composites created and verified: {total_composites_created}")
    print(f"Composites already existing (skipped): {total_composites_skipped}")
    print(f"Total space freed in MinIO s2-products: {total_freed_bytes / (1024**3):.2f} GB ({total_freed_bytes / (1024**4):.2f} TB)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Composite Processing and Raw Data Purge"
    )
    parser.add_argument("--season", type=str, default=None, help="Filter by season (spring, flowering, summer, autumn)")
    parser.add_argument("--tile", type=str, default=None, help="Filter by specific tile (e.g. 30STH)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run without modifying or deleting data")
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "1")), help="Total number of workers in parallel")
    parser.add_argument("--worker-id", type=int, default=int(os.getenv("WORKER_ID", "0")), help="ID of this worker (0 to num-workers - 1)")

    args = parser.parse_args()
    process_composites_and_cleanup(
        season_filter=args.season,
        tile_filter=args.tile,
        dry_run=args.dry_run,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
    )

