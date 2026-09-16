#!/usr/bin/env python3
"""
clean_and_inspect.py - Herramienta de Inspección, Auditoría y Purga Segura para LandCoverPy / MinIO

Uso:
  python clean_and_inspect.py --audit
  python clean_and_inspect.py --create-composites
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
    # Detectar si se ejecuta dentro del contenedor o desde el host
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
    print("🔍 AUDITORÍA DE ALMACENAMIENTO MINIO Y MONGODB")
    print("=" * 70)
    
    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")
    
    # 1. Comprobar existencia de buckets
    for b in [bucket_products, bucket_composites]:
        if not minio_client.bucket_exists(b):
            print(f"⚠️ El bucket '{b}' no existe en MinIO.")
            return

    # 2. Escanear tiles en s2-products
    print(f"\n📦 Analizando bucket '{bucket_products}'...")
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

    print(f"Total objetos raw en MinIO: {total_raw_objects:,}")
    print(f"Espacio total raw en MinIO: {total_raw_bytes / (1024**3):.2f} GB ({total_raw_bytes / (1024**4):.2f} TB)")
    print(f"Tiles detectados con datos raw: {len(tile_stats)}\n")

    # 3. Comprobar composites existentes
    composite_objects = list(minio_client.list_objects(bucket_composites, recursive=True))
    total_comp_bytes = sum(o.size for o in composite_objects)
    print(f"📦 Bucket '{bucket_composites}':")
    print(f"Total objetos composites: {len(composite_objects)}")
    print(f"Espacio composites: {total_comp_bytes / (1024**3):.2f} GB\n")

    # 4. Tabla detallada por tile
    print(f"{'Tile':<10} | {'Objetos':<10} | {'Espacio Raw (GB)':<18} | {'Meses/Estaciones':<25}")
    print("-" * 70)
    for tile in sorted(tile_stats.keys()):
        stat = tile_stats[tile]
        gb = stat["bytes"] / (1024**3)
        seasons_str = ", ".join(sorted(stat["seasons"]))[:24]
        print(f"{tile:<10} | {stat['object_count']:<10} | {gb:<18.2f} | {seasons_str:<25}")

    print("=" * 70)

def purge_raw_products(minio_client, mongo_db, confirm=False):
    print("=" * 70)
    print("🧹 PURGA DE PRODUCTOS RAW EN MINIO")
    print("=" * 70)
    
    bucket_products = os.getenv("MINIO_BUCKET_NAME_PRODUCTS", "s2-products")
    bucket_composites = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")
    
    if not confirm:
        print("⚠️ MODO SIMULACIÓN (DRY-RUN). Usa '--confirm' para ejecutar el borrado real.")
    
    objects = list(minio_client.list_objects(bucket_products, recursive=True))
    total_bytes = sum(o.size for o in objects)
    
    print(f"Se encontraron {len(objects):,} objetos raw en '{bucket_products}' ({total_bytes / (1024**3):.2f} GB).")
    
    if not confirm:
        print("Para purgar estos objetos y liberar espacio de inmediato, ejecuta:")
        print("python clean_and_inspect.py --purge-raw --confirm")
        return

    print("Borrando objetos raw de MinIO...")
    deleted_count = 0
    for obj in objects:
        minio_client.remove_object(bucket_products, obj.object_name)
        deleted_count += 1
        if deleted_count % 500 == 0:
            print(f"Eliminados {deleted_count} / {len(objects)}...")
            
    print(f"✅ Se han eliminado {deleted_count} objetos raw de MinIO.")
    print(f"Espacio liberado en MinIO: ~{total_bytes / (1024**3):.2f} GB.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoría y Limpieza de MinIO para LandCoverPy")
    parser.add_argument("--audit", action="store_true", help="Auditar almacenamiento y estado de tiles")
    parser.add_argument("--purge-raw", action="store_true", help="Purgar productos raw de s2-products")
    parser.add_argument("--confirm", action="store_true", help="Confirmar operaciones destructivas")
    
    args = parser.parse_args()
    
    try:
        minio_cli = get_minio_client()
        mongo_db = get_mongo_client()
    except Exception as e:
        print(f"❌ Error al conectar con MinIO o MongoDB: {e}")
        print("Asegúrate de que MinIO y Mongo están levantados.")
        sys.exit(1)
        
    if args.purge_raw:
        purge_raw_products(minio_cli, mongo_db, confirm=args.confirm)
    else:
        audit_storage(minio_cli, mongo_db)

