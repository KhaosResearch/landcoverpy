import random
import json
from datetime import datetime

from distributed import Client
from glob import glob
from os.path import join
from pathlib import Path

from etc_workflow.execution_mode import ExecutionMode
from etc_workflow.workflow import  _process_tile
from etc_workflow.config import settings
from etc_workflow.model_training import train_model_land_cover
from etc_workflow.utils import (
    get_list_of_tiles_in_iberian_peninsula,
    _connect_mongo_products_collection,
    _connect_mongo_composites_collection,
    _group_polygons_by_tile,
    _get_products_by_tile_and_date,
    get_season_dict,
    _safe_minio_execute,
    _get_minio,
    _create_composite,
    _kmz_to_geojson,
)

def time_composite(client: Client = None):
    if client is not None:
        print("Running in remote dask cluster...")
        future = client.submit(_time_composite)
        execution_time, size_sample = future.result()
    else:
        execution_time, size_sample = _time_composite()

    return execution_time, size_sample

def _time_composite():
    """Returns the execution time of the method `utils._create_composite` using random products."""

    mongo_products = _connect_mongo_products_collection()
    mongo_composites = _connect_mongo_composites_collection()

    minio_client = _get_minio()

    seasons = get_season_dict()

    tiles = get_list_of_tiles_in_iberian_peninsula()
    tile = random.choice(tiles)

    products_available = _get_products_by_tile_and_date(tile, mongo_products, seasons["spring"][0], seasons["autumn"][1], 100)
    products_available = list(products_available)

    size_sample = random.randint(2, 5)
    products_subset = random.sample(products_available, size_sample)

    bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
    bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES


    start_time = datetime.now()
    _create_composite(products_subset, minio_client, bucket_products, bucket_composites, mongo_composites, "training")
    execution_time = datetime.now() - start_time
    return execution_time, size_sample

def time_training_dataset(client: Client = None):
    """Returns the execution time of the method `workflow._process_tile` in a random tile (training)."""

    minio = _get_minio()

    tiles = get_list_of_tiles_in_iberian_peninsula()
    tile = random.choice(tiles)

    geojson_files = []
    for data_class in glob(join(settings.DB_DIR, "*.kmz")):
        if not Path.exists(Path(data_class.replace("kmz","geojson"))):
            print(f"Parsing database to geojson: {data_class}")
            _kmz_to_geojson(data_class)

    for data_class in glob(join(settings.DB_DIR, "*.geojson")):
        print(f"Working with database {data_class}")
        geojson_files.append(data_class)
    polygons_per_tile = _group_polygons_by_tile(*geojson_files)

    metadata_filename = "metadata.json"
    metadata_filepath = join(settings.TMP_DIR, settings.LAND_COVER_MODEL_FOLDER, metadata_filename)

    _safe_minio_execute(
        func=minio.fget_object,
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, settings.LAND_COVER_MODEL_FOLDER, metadata_filename),
        file_path=metadata_filepath,
    )

    with open(metadata_filepath, "r") as metadata_file:
        metadata = json.load(metadata_file)

    used_columns = metadata["used_columns"]

    execution_mode = ExecutionMode.TRAINING

    start_time = datetime.now()

    if client is not None:
        print("Running in remote dask cluster...")
        future = client.submit(_process_tile, tile, execution_mode, polygons_per_tile[tile], used_columns, resources={"Memory": 100})
        client.gather(future)
    else:
        _process_tile(tile, execution_mode, polygons_per_tile[tile], used_columns)

    execution_time = datetime.now() - start_time
    return execution_time, tile

def time_train_model(client: Client = None):
    """Returns the execution time of the method `model_training.train_model_land_cover`."""

    if client is not None:
        print("Running in remote dask cluster...")
        future = client.submit(train_model_land_cover, "dataset_postprocessed.csv", n_jobs = 1, resources={"Memory": 100})
        client.gather(future)
    else:
        train_model_land_cover("dataset_postprocessed.csv", n_jobs = 1)

def time_predicting_tile(client: Client = None):
    """Returns the execution time of the method `workflow._process_tile` in a random tile (predicting)."""

    minio = _get_minio()

    tiles = get_list_of_tiles_in_iberian_peninsula()
    tile = random.choice(tiles)

    print("Predicting tiles")
    polygons_per_tile = {}
    polygons_per_tile[tile] = []

    # For predictions, read the rasters used in "metadata.json".
    metadata_filename = "metadata.json"
    metadata_filepath = join(settings.TMP_DIR, settings.LAND_COVER_MODEL_FOLDER, metadata_filename)

    _safe_minio_execute(
        func=minio.fget_object,
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, settings.LAND_COVER_MODEL_FOLDER, metadata_filename),
        file_path=metadata_filepath,
    )

    with open(metadata_filepath, "r") as metadata_file:
        metadata = json.load(metadata_file)

    used_columns = metadata["used_columns"]

    execution_mode = ExecutionMode.LAND_COVER_PREDICTION

    start_time = datetime.now()

    if client is not None:
        print("Running in remote dask cluster...")
        future = client.submit(_process_tile, tile, execution_mode, polygons_per_tile[tile], used_columns, resources={"Memory": 100})
        client.gather(future)
    else:
        _process_tile(tile, execution_mode, polygons_per_tile[tile], used_columns)

    execution_time = datetime.now() - start_time
    return execution_time, tile

