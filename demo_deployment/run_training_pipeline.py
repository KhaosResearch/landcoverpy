#!/usr/bin/env python3
"""
run_training_pipeline.py

Pipeline resiliente de Extracción de Dataset y Entrenamiento de Modelos de LandCoverPy:
  0. Espera activa/resiliente hasta que el nodo central (.61 MinIO/MongoDB) esté en línea.
  1. Extracción de dataset tabular a partir de los composites estacionales en MinIO (ExecutionMode.TRAINING)
     Reanuda automáticamente saltando los tiles cuyos CSV ya están en MinIO.
  2. Entrenamiento del modelo de cobertura del suelo (RandomForestClassifier - 1er nivel)
  3. Entrenamiento del modelo de subtipos forestales (RandomForestClassifier - 2º nivel para subcategoría 'FOR')
  4. Verificación física de los artefactos subidos a MinIO (datasets/ y ml-models/)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
import urllib3
from pymongo import MongoClient

from landcoverpy.config import settings
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.minio import MinioConnection
from landcoverpy.mongo import MongoConnection
from landcoverpy.model_training import train_model_land_cover, train_second_level_models
from landcoverpy.workflow import workflow
from landcoverpy.utilities.geometries import _csv_to_geojson, _group_validated_data_points_by_tile


def wait_for_central_node():
    """Espera con timeout corto (5s) a que MinIO y MongoDB en .61 respondan (evita bloqueos largos de wslrelay)."""
    while True:
        try:
            # 1. Prueba rápida de salud en MongoDB (timeout 5s)
            fast_mongo = MongoClient(
                host=f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}/",
                username=settings.MONGO_USERNAME,
                password=settings.MONGO_PASSWORD,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
            )
            comp_count = fast_mongo[settings.MONGO_DB][settings.MONGO_COMPOSITES_COLLECTION].count_documents({})

            # 2. Prueba rápida HTTP a MinIO (timeout 5s)
            http = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=5.0, read=5.0),
                retries=urllib3.Retry(total=1)
            )
            resp = http.request("GET", f"http://{settings.MINIO_HOST}:{settings.MINIO_PORT}/minio/health/live")
            if resp.status not in (200, 403, 400):
                raise RuntimeError(f"MinIO health status: {resp.status}")

            minio_client = MinioConnection()
            mongo_client = MongoConnection()
            return minio_client, mongo_client, comp_count
        except Exception as e:
            now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f"[{now_str}] Esperando a que Docker/WSL en {settings.MINIO_HOST} esté operativo ({type(e).__name__}: {e})... Reintentando en 10s.", flush=True)
            time.sleep(10)


def main():
    t_start = time.time()
    print("=" * 80)
    print("  LANDCOVERPY - PIPELINE DE EXTRACCIÓN DE DATASET Y ENTRENAMIENTO DE MODELOS")
    print(f"  Inicio: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80, flush=True)

    # 0. Verificación resiliente de conexiones
    print("\n[0/4] Verificando conexiones a almacenamiento y base de datos...")
    minio_client, mongo_client, comp_count = wait_for_central_node()
    print(f"  * MinIO conectado en {settings.MINIO_HOST}:{settings.MINIO_PORT}")
    print(f"  * MongoDB conectado: {comp_count} composites activos registrados")
    print(f"  * Archivo de inventario: {settings.DB_FILE}")
    print(f"  * Directorio temporal: {settings.TMP_DIR}")
    print(f"  * Carpeta de proyecto MinIO: {settings.MINIO_DATA_FOLDER_NAME}", flush=True)

    # 1. Extracción de dataset de entrenamiento (con bucle de reintento ante cortes de red)
    print("\n" + "=" * 80)
    print("[1/4] FASE 1: Extrayendo reflectancias e índices para puntos de inventario...")
    print("=" * 80, flush=True)
    t1 = time.time()

    data_file = settings.DB_FILE
    if data_file.endswith(".csv"):
        data_file = _csv_to_geojson(data_file, sep=";")
    polygons_per_tile = _group_validated_data_points_by_tile(data_file)

    while True:
        wait_for_central_node()
        try:
            workflow(execution_mode=ExecutionMode.TRAINING, use_aster=False)
            minio_client, _, _ = wait_for_central_node()
            minio_client.stat_object(
                settings.MINIO_BUCKET_DATASETS,
                f"{settings.MINIO_DATA_FOLDER_NAME}/dataset.csv"
            )
            print(f"[OK] Fase 1 completada con éxito en {time.time() - t1:.1f}s", flush=True)
            break
        except Exception as e:
            print(f"[AVISO] Interrupción temporal en Fase 1 ({e}). Reanudando en 10s...", flush=True)
            time.sleep(10)

    # 2. Entrenamiento del modelo principal de cobertura (1er nivel)
    print("\n" + "=" * 80)
    print("[2/4] FASE 2: Entrenando modelo principal de cobertura (Random Forest)...")
    print("=" * 80, flush=True)
    t2 = time.time()
    wait_for_central_node()
    train_model_land_cover(n_jobs=-1)
    print(f"[OK] Fase 2 completada con éxito en {time.time() - t2:.1f}s", flush=True)

    # 3. Entrenamiento de modelos de segundo nivel (subcategorías)
    print("\n" + "=" * 80)
    print("[3/4] FASE 3: Entrenando modelo de 2º nivel para subcategorías (FOR)...")
    print("=" * 80, flush=True)
    t3 = time.time()
    subcategories_env = os.getenv("SUBCATEGORY_PREDICTION", '["FOR"]')
    try:
        subcategories_to_train = list(set(json.loads(subcategories_env)))
    except Exception:
        subcategories_to_train = ["FOR"]

    if subcategories_to_train:
        wait_for_central_node()
        train_second_level_models(subcategories_to_train, n_jobs=-1)
        print(f"[OK] Fase 3 completada con éxito en {time.time() - t3:.1f}s para {subcategories_to_train}", flush=True)

    # 4. Auditoría de artefactos generados en MinIO
    print("\n" + "=" * 80)
    print("[4/4] RESUMEN FINAL: Verificando artefactos generados en MinIO...")
    print("=" * 80, flush=True)
    total_time = time.time() - t_start

    try:
        minio_client, _, _ = wait_for_central_node()
        dataset_objs = list(minio_client.list_objects(
            settings.MINIO_BUCKET_DATASETS,
            prefix=f"{settings.MINIO_DATA_FOLDER_NAME}/",
            recursive=True
        ))
        print(f"\n📦 Bucket '{settings.MINIO_BUCKET_DATASETS}': {len(dataset_objs)} archivos generados")
        for obj in dataset_objs:
            if "tiles_datasets" not in obj.object_name:
                print(f"   - {obj.object_name} ({obj.size / 1024 / 1024:.2f} MB)")

        model_objs = list(minio_client.list_objects(
            settings.MINIO_BUCKET_MODELS,
            prefix=f"{settings.MINIO_DATA_FOLDER_NAME}/",
            recursive=True
        ))
        print(f"\n🧠 Bucket '{settings.MINIO_BUCKET_MODELS}': {len(model_objs)} artefactos generados")
        for obj in model_objs:
            print(f"   - {obj.object_name} ({obj.size / 1024:.1f} KB)")

    except Exception as e:
        print(f"[AVISO] No se pudo listar el resumen final de artefactos: {e}")

    print("\n" + "*" * 80)
    print(f"  ¡ENTRENAMIENTO COMPLETADO CON ÉXITO TOTAL EN {total_time:.1f}s!")
    print("  Los modelos están listos para la fase de predicción ráster distribuida.")
    print("*" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
