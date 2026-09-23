#!/usr/bin/env python3
"""
purge_discarded_indexes.py

Purga retroactiva de indices descartados en MinIO (s2-composites) y MongoDB.
Elimina exclusivamente:
  - evi.tif
  - tci.tif
  - ndwi.tif
  - ndsi.tif

Conserva estrictamente los 11 indices acordados:
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


def get_mongo_col():
    host = os.getenv("MONGO_HOST", "192.168.219.61")
    port = os.getenv("MONGO_PORT", "31114")
    user = os.getenv("MONGO_USERNAME", "adminadmin")
    password = os.getenv("MONGO_PASSWORD", "adminadmin")
    db_name = os.getenv("MONGO_DB", "sentinel2-metadata")
    uri = f"mongodb://{user}:{password}@{host}:{port}/"
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db_name]["composites"]


def purge_discarded_indexes(dry_run: bool = False):
    print("=" * 80)
    print("PURGA RETROACTIVA DE INDICES DESCARTADOS (s2-composites & MongoDB)")
    print(f"Modo: {'SIMULACION (DRY-RUN)' if dry_run else 'EJECUCION REAL'}")
    print(f"Indices a eliminar: {sorted(DISCARDED_INDEXES)}")
    print("=" * 80)

    minio_client = get_minio_client()
    mongo_col = get_mongo_col()
    bucket = os.getenv("MINIO_BUCKET_NAME_COMPOSITES", "s2-composites")

    print(f"\nEscaneando objetos en bucket '{bucket}'...")
    objects = minio_client.list_objects(bucket, recursive=True)

    matched_objects = []
    total_bytes = 0

    for obj in objects:
        filename = obj.object_name.split("/")[-1]
        if filename in DISCARDED_INDEXES and "/indexes/" in obj.object_name:
            matched_objects.append(obj)
            total_bytes += (obj.size or 0)

    print(f"Detectados {len(matched_objects)} archivos descartados a purgar.")
    print(f"Espacio total a liberar: {total_bytes / (1024**3):.2f} GB ({total_bytes / (1024**2):.2f} MB)")

    if dry_run:
        print("\n[DRY-RUN] No se han realizado cambios en MinIO ni en MongoDB.")
        return

    if not matched_objects:
        print("No se encontraron archivos pendientes de purga.")
        return

    print("\nEliminando objetos en MinIO...")
    deleted_count = 0
    t0 = time.time()

    for idx, obj in enumerate(matched_objects, 1):
        try:
            minio_client.remove_object(bucket, obj.object_name)
            deleted_count += 1
            if idx % 100 == 0 or idx == len(matched_objects):
                pct = (idx / len(matched_objects)) * 100
                print(f"  Progreso: {idx}/{len(matched_objects)} ({pct:.1f}%) eliminados...")
        except Exception as e:
            print(f"  [ERROR] Fallo al eliminar {obj.object_name}: {e}")

    elapsed = time.time() - t0
    print(f"\nPurga en MinIO completada en {elapsed:.1f}s.")
    print(f"Total archivos eliminados: {deleted_count}/{len(matched_objects)}")
    print(f"Espacio liberado en MinIO: {total_bytes / (1024**3):.2f} GB")

    print("\nActualizando metadatos en MongoDB (removiendo claves descartadas)...")
    res = mongo_col.update_many({}, {"$unset": DISCARDED_MONGO_KEYS})
    print(f"Documentos modificados en MongoDB: {res.modified_count}")

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purga de indices descartados en MinIO y MongoDB")
    parser.add_argument("--dry-run", action="store_true", help="Simulacion sin modificar datos")
    args = parser.parse_args()

    purge_discarded_indexes(dry_run=args.dry_run)
