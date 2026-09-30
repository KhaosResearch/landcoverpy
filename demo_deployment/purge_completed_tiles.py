#!/usr/bin/env python3
"""
purge_completed_tiles.py

Retroactive cleanup utility for Sentinel-2 in MinIO and MongoDB.
Identifies composites that are already completed and verified in s2-composites,
and completely removes all associated raw captures, intermediate products (cloud masks),
and residual cloudy scenes from s2-products to immediately reclaim storage.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pymongo
from minio import Minio


from minio.deleteobjects import DeleteObject

SEASON_MONTHS = {
    "spring": ["March", "April"],
    "flowering": ["May"],
    "summer": ["June", "July"],
    "autumn": ["October", "November"],
}


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


def run_purge(dry_run: bool = True, season_filter: str = None, tile_filter: str = None):
    minio_client = get_minio_client()
    db = get_mongo_db()
    
    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")

    seasons_file = os.getenv("SEASONS_FILE", "demo_deployment/app_data/seasons.json")
    if not Path(seasons_file).exists():
        seasons_file = "/app/data/seasons.json"
    if not Path(seasons_file).exists():
        seasons_file = "app_data/seasons.json"

    with open(seasons_file, "r") as f:
        seasons_data = json.load(f)

    print("=" * 80)
    print("RETROACTIVE RAW PRODUCTS PURGE FOR COMPLETED TILES")
    print(f"Mode: {'[DRY-RUN / SIMULATION]' if dry_run else '[LIVE RUN - DELETING OBJECTS]'}")
    if season_filter:
        print(f"Season filter: {season_filter.upper()}")
    if tile_filter:
        print(f"Tile filter: {tile_filter}")
    print("=" * 80)

    # Find completed composites in MongoDB
    query = {}
    if season_filter:
        query["season"] = season_filter.lower()
    if tile_filter:
        query["tile"] = tile_filter

    completed_composites = list(db.composites.find(query, {"tile": 1, "season": 1}))
    print(f"\nTotal completed composites found in MongoDB: {len(completed_composites)}")

    total_deleted_objects = 0
    total_deleted_bytes = 0
    tiles_purged = 0

    for idx, comp in enumerate(completed_composites, 1):
        tile = comp.get("tile")
        season = comp.get("season")
        if not tile or not season:
            continue

        if season not in seasons_data:
            continue

        s_info = seasons_data[season]
        s_start = datetime.strptime(s_info["start"], "%Y-%m-%d")
        s_end = datetime.strptime(s_info["end"], "%Y-%m-%d")
        months = SEASON_MONTHS.get(season, [])

        # Ensure composite actually exists in MinIO before purging any raw data
        comp_objs = list(minio_client.list_objects(bucket_composites, prefix=f"{tile}/2021/{season}/", recursive=True))
        if len(comp_objs) < 10:
            print(f"  [{idx}/{len(completed_composites)}: {tile}] [{season}] Incomplete composite in MinIO ({len(comp_objs)} layers). Skipping for safety.")
            continue

        tile_deleted_objs = 0
        tile_deleted_bytes = 0

        # Method 1: Delete folders by month for the season (tile/2021/Month/)
        # This removes 100% of raw/, intermediateProducts/, and residual/orphaned cloudy captures
        for month in months:
            month_prefix = f"{tile}/2021/{month}/"
            try:
                raw_objs = list(minio_client.list_objects(bucket_products, prefix=month_prefix, recursive=True))
                if raw_objs:
                    for obj in raw_objs:
                        tile_deleted_bytes += obj.size or 0
                        tile_deleted_objs += 1
                    if not dry_run:
                        del_list = [DeleteObject(o.object_name) for o in raw_objs]
                        for c_i in range(0, len(del_list), 1000):
                            del_chunk = del_list[c_i:c_i + 1000]
                            errors = list(minio_client.remove_objects(bucket_products, del_chunk))
                            if errors:
                                for err in errors:
                                    print(f"  [ERROR] Deleting {err.name}: {err.message}")
            except Exception as e:
                print(f"  [ERROR] Scanning/deleting {month_prefix} in MinIO: {e}")

        # Method 2: Clean heavy metadata in MongoDB for captures in that tile/period
        if not dry_run and tile_deleted_objs > 0:
            try:
                db.products.update_many(
                    {
                        "tile": tile,
                        "datetakeSensingTime": {"$gte": s_start, "$lt": s_end}
                    },
                    {"$unset": {"indexes": "", "intermediateProducts": ""}}
                )
            except Exception as e:
                print(f"  [ERROR] Cleaning MongoDB for {tile} ({season}): {e}")

        if tile_deleted_objs > 0:
            tiles_purged += 1
            action = "Purged" if not dry_run else "Would purge"
            print(f"  [{idx}/{len(completed_composites)}: {tile}] [{season}] {action} {tile_deleted_objs} raw/intermediate objects (+{tile_deleted_bytes / (1024**3):.2f} GB).")

        total_deleted_objects += tile_deleted_objs
        total_deleted_bytes += tile_deleted_bytes

    print("\n" + "=" * 80)
    print("PURGE SUMMARY")
    print(f"Tiles processed with raw data purged: {tiles_purged}")
    print(f"Total raw/intermediate objects deleted: {total_deleted_objects}")
    print(f"Total space freed in MinIO s2-products: {total_deleted_bytes / (1024**3):.2f} GB ({total_deleted_bytes / (1024**4):.2f} TB)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge raw products for completed composites")
    parser.add_argument("--confirm", action="store_true", help="Execute actual deletion (defaults to dry-run)")
    parser.add_argument("--season", type=str, default=None, help="Filter by season (spring, flowering, summer, autumn)")
    parser.add_argument("--tile", type=str, default=None, help="Filter by specific tile (e.g. 30STH)")
    
    args = parser.parse_args()
    run_purge(dry_run=not args.confirm, season_filter=args.season, tile_filter=args.tile)
