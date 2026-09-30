#!/usr/bin/env python3
"""
clean_and_inspect.py - Inspection, Audit, and Safe Purge Utility for LandCoverPy / MinIO

Usage:
  python clean_and_inspect.py --audit
  python clean_and_inspect.py --purge-raw --confirm
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from minio import Minio
from minio.error import S3Error
import pymongo

def get_minio_client():
    # Detect execution within container or from host
    host = os.getenv("MINIO_HOST", "localhost")
    port = os.getenv("MINIO_PORT", "9000" if host != "localhost" else "31113")
    access_key = os.getenv("MINIO_ACCESS_KEY", "adminadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "adminadmin")
    
    return Minio(
        f"{host}:{port}",
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

def get_mongo_client():
    host = os.getenv("MONGO_HOST", "localhost")
    port = os.getenv("MONGO_PORT", "27017" if host != "localhost" else "31114")
    user = os.getenv("MONGO_USERNAME", "adminadmin")
    password = os.getenv("MONGO_PASSWORD", "adminadmin")
    db_name = os.getenv("MONGO_DB", "sentinel2-metadata")
    
    uri = f"mongodb://{user}:{password}@{host}:{port}/"
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db_name]

def audit_storage(minio_client, mongo_db):
    print("=" * 70)
    print("MINIO AND MONGODB STORAGE AUDIT")
    print("=" * 70)
    
    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")
    
    # 1. Check bucket existence
    for b in [bucket_products, bucket_composites]:
        if not minio_client.bucket_exists(b):
            print(f"Warning: Bucket '{b}' does not exist in MinIO.")
            return

    # 2. Scan tiles in s2-products
    print(f"\nAnalyzing bucket '{bucket_products}'...")
    products_col = mongo_db["products"]
    composites_col = mongo_db["composites"]

    tile_stats = {}
    objects = minio_client.list_objects(bucket_products, recursive=True)
    
    total_raw_bytes = 0
    total_raw_objects = 0
    
    for obj in objects:
        total_raw_bytes += obj.size
        total_raw_objects += 1
        parts = obj.object_name.split("/")
        if len(parts) > 0:
            tile = parts[0]
            if tile not in tile_stats:
                tile_stats[tile] = {
                    "object_count": 0,
                    "bytes": 0,
                    "seasons": set()
                }
            tile_stats[tile]["object_count"] += 1
            tile_stats[tile]["bytes"] += obj.size
            if len(parts) >= 3:
                # tile/year/month
                tile_stats[tile]["seasons"].add(f"{parts[1]}/{parts[2]}")

    print(f"Total raw objects in MinIO: {total_raw_objects:,}")
    print(f"Total raw storage in MinIO: {total_raw_bytes / (1024**3):.2f} GB ({total_raw_bytes / (1024**4):.2f} TB)")
    print(f"Tiles detected with raw data: {len(tile_stats)}\n")

    # 3. Check existing composites
    composite_objects = list(minio_client.list_objects(bucket_composites, recursive=True))
    total_comp_bytes = sum(o.size for o in composite_objects)
    print(f"Bucket '{bucket_composites}':")
    print(f"Total composite objects: {len(composite_objects)}")
    print(f"Composite storage: {total_comp_bytes / (1024**3):.2f} GB\n")

    # 4. Detailed tile table
    print(f"{'Tile':<10} | {'Objects':<10} | {'Raw Space (GB)':<18} | {'Months/Seasons':<25}")
    print("-" * 70)
    for tile in sorted(tile_stats.keys()):
        stat = tile_stats[tile]
        gb = stat["bytes"] / (1024**3)
        seasons_str = ", ".join(sorted(stat["seasons"]))[:24]
        print(f"{tile:<10} | {stat['object_count']:<10} | {gb:<18.2f} | {seasons_str:<25}")

    print("=" * 70)

def purge_raw_products(minio_client, mongo_db, confirm=False):
    print("=" * 70)
    print("PURGING RAW PRODUCTS IN MINIO")
    print("=" * 70)
    
    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")
    
    if not confirm:
        print("SIMULATION MODE (DRY-RUN). Use '--confirm' to execute actual deletion.")
    
    objects = list(minio_client.list_objects(bucket_products, recursive=True))
    total_bytes = sum(o.size for o in objects)
    
    print(f"Found {len(objects):,} raw objects in '{bucket_products}' ({total_bytes / (1024**3):.2f} GB).")
    
    if not confirm:
        print("To purge these objects and reclaim storage immediately, run:")
        print("python clean_and_inspect.py --purge-raw --confirm")
        return

    print("Deleting raw objects from MinIO...")
    deleted_count = 0
    for obj in objects:
        minio_client.remove_object(bucket_products, obj.object_name)
        deleted_count += 1
        if deleted_count % 500 == 0:
            print(f"Deleted {deleted_count} / {len(objects)}...")
            
    print(f"Deleted {deleted_count} raw objects from MinIO.")
    print(f"Storage freed in MinIO: ~{total_bytes / (1024**3):.2f} GB.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinIO Storage Audit and Cleanup for LandCoverPy")
    parser.add_argument("--audit", action="store_true", help="Audit storage and tile status")
    parser.add_argument("--purge-raw", action="store_true", help="Purge raw products from s2-products")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive purge operations")
    
    args = parser.parse_args()
    
    try:
        minio_cli = get_minio_client()
        mongo_db = get_mongo_client()
    except Exception as e:
        print(f"Error connecting to MinIO or MongoDB: {e}")
        print("Ensure MinIO and Mongo services are running and accessible.")
        sys.exit(1)
        
    if args.purge_raw:
        purge_raw_products(minio_cli, mongo_db, confirm=args.confirm)
    else:
        audit_storage(minio_cli, mongo_db)

