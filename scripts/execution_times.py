import random
import json
from datetime import datetime
from os.path import join
from pathlib import Path

from distributed import Client
from glob import glob

from landcoverpy.composite import _create_composite
from landcoverpy.config import settings
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.model_training import train_model_land_cover
from landcoverpy.mongo import MongoConnection
from landcoverpy.minio import MinioConnection
from landcoverpy.utilities.aoi_tiles import get_list_of_tiles_in_iberian_peninsula
from landcoverpy.utilities.geometries import _group_validated_data_points_by_tile, _kmz_to_geojson, _csv_to_geojson
from landcoverpy.utilities.utils import get_products_by_tile_and_date, get_season_dict
from landcoverpy.workflow import  _process_tile

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

    mongo_client = MongoConnection()
    mongo_products = mongo_client.get_collection_object()

    minio_client = MinioConnection()

    seasons = get_season_dict()

    tiles = get_list_of_tiles_in_iberian_peninsula()
    tile = random.choice(tiles)

    start_date = seasons[seasons.keys()[0]][0]
    end_date = seasons[seasons.keys()[-1]][1]
    products_available = get_products_by_tile_and_date(tile, mongo_products, start_date, end_date, 100)
    products_available = list(products_available)

    size_sample = random.randint(2, 5)
    products_subset = random.sample(products_available, size_sample)

    bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
    bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES

    mongo_client.set_collection(settings.MONGO_COMPOSITES_COLLECTION)
    mongo_composites = mongo_client.get_collection_object()

    start_time = datetime.now()
    _create_composite(products_subset, minio_client, bucket_products, bucket_composites, mongo_composites, "training")
    execution_time = datetime.now() - start_time
    return execution_time, size_sample

def time_training_dataset(client: Client = None):
    """Returns the execution time of the method `workflow._process_tile` in a random tile (training)."""

    minio = MinioConnection()

    tiles = get_list_of_tiles_in_iberian_peninsula()
    tile = random.choice(tiles)

    data_file = settings.DB_FILE
    if data_file.endswith(".kmz"):
        data_file = _kmz_to_geojson(data_file)
    if data_file.endswith(".csv"):
        data_file = _csv_to_geojson(data_file, sep=',')

    polygons_per_tile = _group_validated_data_points_by_tile(data_file)

    metadata_filename = "metadata.json"
    metadata_filepath = join(settings.TMP_DIR, "land-cover", metadata_filename)

    minio.fget_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, "land-cover", metadata_filename),
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

    minio = MinioConnection()

    tiles = get_list_of_tiles_in_iberian_peninsula()
    tile = random.choice(tiles)

    print("Predicting tiles")
    polygons_per_tile = {}
    polygons_per_tile[tile] = []

    # For predictions, read the rasters used in "metadata.json".
    metadata_filename = "metadata.json"
    metadata_filepath = join(settings.TMP_DIR, "land-cover", metadata_filename)

    minio.fget_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, "land-cover", metadata_filename),
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

