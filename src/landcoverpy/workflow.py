import json
from os.path import join
from typing import List

import pandas as pd
from distributed import Client

from landcoverpy.config import settings
from landcoverpy.exceptions import WorkflowExecutionException
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.minio import MinioConnection
from landcoverpy.utilities.geometries import _group_validated_data_points_by_tile, _kmz_to_geojson, _csv_to_geojson
from landcoverpy.utilities.utils import (
    _check_tiles_not_predicted,
    _remove_tiles_already_processed_in_training,
)

from landcoverpy.workflow_predict import _process_tile_predict
from landcoverpy.workflow_train import _process_tile_train

def workflow(
        execution_mode: ExecutionMode, 
        client: Client = None, 
        tiles_to_predict: List[str] = None,
        use_block_windows: bool = False,
        window_slices: tuple = None):

    if execution_mode == ExecutionMode.TRAINING and tiles_to_predict is not None:
        print("Warning: tiles_to_predict are ignored in training mode.")
    if execution_mode == ExecutionMode.TRAINING and (use_block_windows or window_slices is not None):
        print("Warning: Windows are ignored in training mode, since it is already optimized for low memory usage.")
    elif window_slices is not None and use_block_windows:
        print("Warning: if use_block_windows is enabled, window_slices will be ignored.")

    predict = execution_mode != ExecutionMode.TRAINING

    minio = MinioConnection()

    data_file = settings.DB_FILE
    if data_file.endswith(".kmz"):
        data_file = _kmz_to_geojson(data_file)
    if data_file.endswith(".csv"):
        data_file = _csv_to_geojson(data_file, sep=',')

    polygons_per_tile = _group_validated_data_points_by_tile(data_file)

    if predict:
        print("Predicting tiles")
        if tiles_to_predict is not None:
            polygons_per_tile = {}
            for tile_to_predict in tiles_to_predict:
                polygons_per_tile[tile_to_predict] = []

        tiles = _check_tiles_not_predicted(list(polygons_per_tile.keys()))

        if execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:
            sl_tiles = _check_tiles_not_predicted(list(polygons_per_tile.keys()), second_level_prediction=True)

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

    else:
        print("Creating dataset from tiles")
        # Tiles related to the traininig zone that wasn't already processed
        tiles = _remove_tiles_already_processed_in_training(list(polygons_per_tile.keys()))

        # In training, read all rasters available
        used_columns = None

    if client is not None:
        futures = []
        if execution_mode == ExecutionMode.TRAINING:
            for tile in tiles:
                future = client.submit(_process_tile_train, tile, polygons_per_tile[tile], resources={"Memory": 100})
                futures.append(future)
        elif execution_mode == ExecutionMode.LAND_COVER_PREDICTION or execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:
            for tile in tiles:
                future = client.submit(_process_tile_predict, tile, ExecutionMode.LAND_COVER_PREDICTION, used_columns, use_block_windows, window_slices, resources={"Memory": 100})
                futures.append(future)
            client.gather(futures, errors="skip")
        elif execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:
            for tile in sl_tiles:
                future = client.submit(_process_tile_predict, tile, ExecutionMode.SECOND_LEVEL_PREDICTION, used_columns, use_block_windows, window_slices, resources={"Memory": 100})
                futures.append(future)
        client.gather(futures, errors="skip")

    else:
        if execution_mode == ExecutionMode.TRAINING:
            for tile in tiles:
                try:
                    _process_tile_train(tile, polygons_per_tile[tile])
                except WorkflowExecutionException as e:
                    print(e)
        if execution_mode == ExecutionMode.LAND_COVER_PREDICTION or execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:
            for tile in tiles:
                try:
                    _process_tile_predict(tile, ExecutionMode.LAND_COVER_PREDICTION, used_columns, use_block_windows, window_slices)
                except WorkflowExecutionException as e:
                    print(e)
        if execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:
            for tile in sl_tiles:
                try:
                    _process_tile_predict(tile, ExecutionMode.SECOND_LEVEL_PREDICTION, used_columns, use_block_windows, window_slices)
                except WorkflowExecutionException as e:
                    print(e)


    if not predict:
        # Merge all tiles datasets into a big dataset.csv, then upload it to minio

        final_df = None

        minio = MinioConnection()

        tiles_datasets_minio_dir = join(
            settings.MINIO_DATA_FOLDER_NAME, "tiles_datasets", ""
        )
        tiles_datasets_minio_cursor = minio.list_objects(
            bucket_name=settings.MINIO_BUCKET_DATASETS,
            prefix=tiles_datasets_minio_dir,
        )

        local_dataset_path = join(settings.TMP_DIR, "tile_dataset.csv")

        for tile_dataset_minio_object in tiles_datasets_minio_cursor:

            tile_dataset_minio_path = tile_dataset_minio_object.object_name

            minio.fget_object(
                bucket_name=settings.MINIO_BUCKET_DATASETS,
                object_name=tile_dataset_minio_path,
                file_path=local_dataset_path,
            )

            tile_df = pd.read_csv(local_dataset_path)

            if final_df is None:
                final_df = tile_df
            else:
                final_df = pd.concat([final_df, tile_df], axis=0)

        print(final_df)
        file_path = join(settings.TMP_DIR, "dataset.csv")
        final_df.to_csv(file_path, index=False)
        minio.fput_object(
            bucket_name=settings.MINIO_BUCKET_DATASETS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, "dataset.csv"),
            file_path=file_path,
            content_type="text/csv",
        )