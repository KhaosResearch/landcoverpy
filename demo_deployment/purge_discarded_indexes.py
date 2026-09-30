#!/usr/bin/env python3
"""
purge_discarded_indexes.py

Retroactive cleanup utility for discarded indices in MinIO (s2-composites) and MongoDB.
Exclusively removes:
  - evi.tif
  - tci.tif
  - ndwi.tif
  - ndsi.tif

Strictly preserves the 11 approved seasonal indices:
  - cri1.tif, ri.tif, evi2.tif, mndwi.tif, moisture.tif
  - ndyi.tif, ndre.tif, ndvi.tif, osavi.tif, bri.tif, bsi.tif
"""

import argparse
import os
import sys
import time
from typing import List, Set

from minio import Minio
import pymongo


DISCARDED_INDEXES = {
    "evi.tif",
    "tci.tif",
    "ndwi.tif",
    "ndsi.tif",
}

DISCARDED_MONGO_KEYS = {
    "indexes.evi": "",
    "indexes.tci": "",
    "indexes.ndwi": "",
    "indexes.ndsi": "",
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


def get_mongo_col():
    host = os.getenv("MONGO_HOST", "localhost")
    port = os.getenv("MONGO_PORT", "31114")
    user = os.getenv("MONGO_USERNAME", "adminadmin")
    password = os.getenv("MONGO_PASSWORD", "adminadmin")
    db_name = os.getenv("MONGO_DB", "sentinel2-metadata")
    uri = f"mongodb://{user}:{password}@{host}:{port}/"
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db_name]["composites"]


def purge_discarded_indexes(dry_run: bool = False):
    print("=" * 80)
    print("RETROACTIVE PURGE OF DISCARDED INDICES (s2-composites & MongoDB)")
    print(f"Mode: {'SIMULATION (DRY-RUN)' if dry_run else 'LIVE RUN'}")
    print(f"Indices to remove: {sorted(DISCARDED_INDEXES)}")
    print("=" * 80)

    minio_client = get_minio_client()
    mongo_col = get_mongo_col()
    bucket = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")

    print(f"\nScanning objects in bucket '{bucket}'...")
    objects = minio_client.list_objects(bucket, recursive=True)

    matched_objects = []
    total_bytes = 0

    for obj in objects:
        filename = obj.object_name.split("/")[-1]
        if filename in DISCARDED_INDEXES and "/indexes/" in obj.object_name:
            matched_objects.append(obj)
            total_bytes += (obj.size or 0)

    print(f"Detected {len(matched_objects)} discarded files to purge.")
    print(f"Total space to reclaim: {total_bytes / (1024**3):.2f} GB ({total_bytes / (1024**2):.2f} MB)")

    if dry_run:
        print("\n[DRY-RUN] No changes were made in MinIO or MongoDB.")
        return

    if not matched_objects:
        print("No files pending purge were found.")
        return

    print("\nDeleting objects in MinIO...")
    deleted_count = 0
    t0 = time.time()

    for idx, obj in enumerate(matched_objects, 1):
        try:
            minio_client.remove_object(bucket, obj.object_name)
            deleted_count += 1
            if idx % 100 == 0 or idx == len(matched_objects):
                pct = (idx / len(matched_objects)) * 100
                print(f"  Progress: {idx}/{len(matched_objects)} ({pct:.1f}%) deleted...")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {obj.object_name}: {e}")

    elapsed = time.time() - t0
    print(f"\nMinIO purge completed in {elapsed:.1f}s.")
    print(f"Total files deleted: {deleted_count}/{len(matched_objects)}")
    print(f"Space freed in MinIO: {total_bytes / (1024**3):.2f} GB")

    print("\nUpdating metadata in MongoDB (unsetting discarded keys)...")
    res = mongo_col.update_many({}, {"$unset": DISCARDED_MONGO_KEYS})
    print(f"Documents modified in MongoDB: {res.modified_count}")

    print("\n" + "=" * 80)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge discarded spectral indices in MinIO and MongoDB")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run without modifying data")
    args = parser.parse_args()

    purge_discarded_indexes(dry_run=args.dry_run)

