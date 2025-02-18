import json
from collections import defaultdict
from os.path import join
from pathlib import Path
from shutil import rmtree

import joblib
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from rasterio.windows import Window

from landcoverpy.aster import get_dem_from_tile
from landcoverpy.composite import _create_composite, _get_composite, _validate_composite_products
from landcoverpy.config import settings
from landcoverpy.exceptions import WorkflowExecutionException, NoSentinelException
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.minio import MinioConnection
from landcoverpy.mongo import MongoConnection
from landcoverpy.utilities.raster import (
    _download_sample_band_by_tile,
    _filter_rasters_paths_by_features_used,
    _get_kwargs_raster,
    _get_product_rasters_paths,
    _get_raster_filename_from_path,
    _get_raster_name_from_path,
    _read_raster,
    _get_block_windows_by_tile,
    _generate_windows_from_slices_number,
)
from landcoverpy.utilities.utils import (
    _get_lc_classification,
    get_products_by_tile_and_date,
    get_season_dict,
)

def _process_tile_predict(tile, execution_mode, used_columns=None, use_block_windows=False, window_slices=None, use_aster=True):
    if execution_mode == ExecutionMode.TRAINING:
        raise WorkflowExecutionException("This function is only for prediction")
    
    print(f"Predicting tile {tile}")

    if not Path(settings.TMP_DIR).exists():
        Path.mkdir(Path(settings.TMP_DIR))

    seasons = get_season_dict()

    minio_client = MinioConnection()
    mongo_client = MongoConnection()
    mongo_products_collection = mongo_client.get_collection_object()

    model_name = "model.joblib"
    lc_model_folder = "land-cover"

    with open(settings.LC_LABELS_FILE, "r") as f:
        lc_mapping = json.load(f)
    
    if 0 in lc_mapping.values():
        print("Warning: 0 is already a value in the mapping, which is reserved for NODATA. It will be overwritten.")
    lc_mapping["nodata"] = 0

    if execution_mode == ExecutionMode.LAND_COVER_PREDICTION:
        model_path = join(settings.TMP_DIR, lc_model_folder, model_name)

        print("Downloading land cover model")
        minio_client.fget_object(
            bucket_name=settings.MINIO_BUCKET_MODELS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, lc_model_folder, model_name),
            file_path=model_path,
        )

        lc_clf = joblib.load(model_path)

    elif execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:

        model_folders = minio_client.list_objects(settings.MINIO_BUCKET_MODELS, prefix=join(settings.MINIO_DATA_FOLDER_NAME, ''), recursive=False)
        sl_model_folder = []
        for model_folder in model_folders:
            if model_folder.object_name.endswith('/') and lc_model_folder not in model_folder.object_name:
                sl_model_folder.append(model_folder.object_name.split('/')[-2])

        local_sl_model_locations = {}
        for minio_model_folder in sl_model_folder:

            local_sl_model_path = join(settings.TMP_DIR, minio_model_folder, model_name)

            print(f"Downloading {minio_model_folder}/{model_name} model")
            minio_client.fget_object(
                bucket_name=settings.MINIO_BUCKET_MODELS,
                object_name=join(settings.MINIO_DATA_FOLDER_NAME, minio_model_folder, model_name),
                file_path=local_sl_model_path,
            )
            local_sl_model_locations[minio_model_folder] = local_sl_model_path

        with open(settings.SL_LABELS_FILE, "r") as f:
            sl_mapping = json.load(f)

        if 0 in sl_mapping.values():
            print("Warning: 0 is already a value in the mapping, which is reserved for NODATA. It will be overwritten.")
        sl_mapping["nodata"] = 0
        if 1 in sl_mapping.values():
            print("Warning: 1 is already a value in the mapping, which is reserved for NOCLASSIFIED. It will be overwritten.")
        sl_mapping["noclassified"] = 1

        sl_classifiers = {}

        for sl_model in local_sl_model_locations.keys():
            sl_classifiers[lc_mapping[sl_model]] = joblib.load(local_sl_model_locations[sl_model])
    

    band_path = _download_sample_band_by_tile(tile, minio_client, mongo_products_collection)
    kwargs_s2 = _get_kwargs_raster(band_path)

    if use_block_windows:
        windows = _get_block_windows_by_tile(tile, minio_client, mongo_products_collection)
    elif window_slices is not None:
        windows = _generate_windows_from_slices_number(tile, window_slices, minio_client, mongo_products_collection)
    else:
        windows = [Window(0, 0, kwargs_s2['width'], kwargs_s2['height'])]

    # Initialize empty output raster
    output_kwargs = kwargs_s2.copy()
    output_kwargs["nodata"] = 0
    output_kwargs["driver"] = "GTiff"
    output_kwargs["dtype"] = np.uint8
    if execution_mode == ExecutionMode.LAND_COVER_PREDICTION:
        classification_name = f"classification_{tile}.tif"
    elif execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:
        classification_name = f"sl_classification_{tile}.tif"
    classification_path = str(Path(settings.TMP_DIR, classification_name))
    with rasterio.open(
        classification_path, "w", **output_kwargs
    ) as classification_file:
        print(f"Initilized output raster at {classification_path}")


    # Names of the indexes that are taken into account
    indexes_used = [
        "cri1",
        "ri",
        "evi2",
        "mndwi",
        "moisture",
        "ndyi",
        "ndre",
        "ndvi",
        "osavi",
    ]
    # Name of the sentinel bands that are ignored
    skip_bands = ["tci", "scl"]
    # Ranges for normalization of each raster
    normalize_range = {"slope": (0, 70), "aspect": (0, 360), "dem": (0, 2000)}

    print(f"Working in tile {tile}")
    # Mongo query for obtaining valid products
    min_useful_data_percentage = settings.MIN_USEFUL_DATA_PERCENTAGE

    product_per_season = {}

    for season in seasons:
        season_start, season_end = seasons[season]
        product_metadata_cursor = get_products_by_tile_and_date(
            tile, mongo_products_collection, season_start, season_end, min_useful_data_percentage
        )

        product_metadata_list = list(product_metadata_cursor)
        #filter corrupted products
        product_metadata_list = _validate_composite_products(product_metadata_list)
        # If there are more products than the maximum specified for creating a composite, take the last ones
        product_per_season[season] = product_metadata_list[:settings.MAX_PRODUCTS_COMPOSITE]

        if len(product_per_season[season]) == 0:
            raise NoSentinelException(f"There is no valid Sentinel products for tile {tile} in season {season}. Skipping it...")


    rasters_by_season = defaultdict(dict)

    for season, products_metadata in product_per_season.items():

        if len(products_metadata) == 0:
            raise NoSentinelException(f"There is no valid Sentinel products for tile {tile}. Skipping it...")
        else:
            products_metadata_list = list(products_metadata)
            product_metadata = _get_composite(
                products_metadata_list, execution_mode
            )
            if product_metadata is None:
                _create_composite(
                    products_metadata_list,
                    execution_mode
                )
                product_metadata = _get_composite(
                    products_metadata_list, execution_mode
                )

        product_name = product_metadata["title"]

        minio_bucket = product_metadata["S3Bucket"]

        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client
        )

        (rasters_paths, is_band) = _filter_rasters_paths_by_features_used(
            rasters_paths, is_band, used_columns, season
        )
        temp_product_folder = Path(settings.TMP_DIR, product_name + ".SAFE")
        if not temp_product_folder.exists():
            Path.mkdir(temp_product_folder)

        rasters_by_season[season]["raster_paths"] = rasters_paths
        rasters_by_season[season]["is_band"] = is_band
        rasters_by_season[season]["temp_product_folder"] = temp_product_folder


    if use_aster:
        dems_raster_names = [
            "slope",
            "aspect",
            "dem",
        ]
    else:
        dems_raster_names = []


    dem_paths = {}
    for dem_name in dems_raster_names:
        # Add dem and aspect data
        if dem_name in used_columns:
            dem_path = get_dem_from_tile(
                execution_mode, tile, mongo_products_collection, minio_client, dem_name
            )
            dem_paths[dem_name] = dem_path


    for window in windows:
        print(f"Processing window {window}")
        window_kwargs = kwargs_s2.copy()
        window_kwargs["width"] = window.width
        window_kwargs["height"] = window.height
        window_kwargs["transform"] = rasterio.windows.transform(window, kwargs_s2["transform"])

        # Dict for storing data of a window (all rasters and indexes)
        window_tile_dict = {}

        crop_mask = np.zeros(shape=(int(window_kwargs["height"]), int(window_kwargs["width"])), dtype=np.uint8)

        for season in seasons:
            # Read bands and indexes.
            already_read = []
            for i, raster_path in enumerate(rasters_by_season[season]["raster_paths"]):
                raster_filename = _get_raster_filename_from_path(raster_path)
                raster_name = _get_raster_name_from_path(raster_path)
                temp_path = Path(rasters_by_season[season]["temp_product_folder"], raster_filename)

                # Only keep bands and indexes in indexes_used
                if (not rasters_by_season[season]["is_band"][i]) and (
                    not any(
                        raster_name.upper() == index_used.upper()
                        for index_used in indexes_used
                    )
                ):
                    continue
                # Skip bands in skip_bands
                if rasters_by_season[season]["is_band"][i] and any(
                    raster_name.upper() == band_skipped.upper()
                    for band_skipped in skip_bands
                ):
                    continue
                # Read only the first band to avoid duplication of different spatial resolution
                if any(
                    raster_name.upper() == read_raster.upper()
                    for read_raster in already_read
                ):
                    continue
                already_read.append(raster_name)

                # download if file_path does not exist
                if not temp_path.exists():
                    print(f"Downloading raster {raster_name} from minio into {temp_path}")
                    minio_client.fget_object(
                        bucket_name=minio_bucket,
                        object_name=raster_path,
                        file_path=str(temp_path),
                    )

                band_normalize_range = normalize_range.get(raster_name, None)
                if rasters_by_season[season]["is_band"][i] and (band_normalize_range is None):
                    band_normalize_range = (0, 7000)

                raster = _read_raster(
                    band_path=temp_path,
                    rescale=True,
                    normalize_range=band_normalize_range,
                    window=window
                )
                raster_masked = np.ma.masked_array(raster[0], mask=crop_mask)
                raster_masked = np.ma.compressed(raster_masked)

                window_tile_dict.update({f"{season}_{raster_name}": raster_masked})

        for dem_name in dems_raster_names:
            # Add dem and aspect data
            if dem_name in used_columns:

                # Predict doesnt work yet in some specific tiles where tile is not fully contained in aster rasters
                dem_kwargs = _get_kwargs_raster(dem_paths[dem_name])

                dem_kwargs_window = dem_kwargs.copy()
                dem_kwargs_window["width"] = window.width
                dem_kwargs_window["height"] = window.height
                dem_kwargs_window["transform"] = rasterio.windows.transform(window, dem_kwargs["transform"])

                crop_mask = np.zeros(shape=(int(dem_kwargs_window["height"]), int(dem_kwargs_window["width"])),dtype=np.uint8)

                band_normalize_range = normalize_range.get(dem_name, None)
                raster = _read_raster(
                    band_path=dem_paths[dem_name],
                    rescale=True,
                    normalize_range=band_normalize_range,
                    window=window
                )
                raster_masked = np.ma.masked_array(raster, mask=crop_mask)
                raster_masked = np.ma.compressed(raster_masked).flatten()
                window_tile_dict.update({dem_name: raster_masked})

        window_tile_df = pd.DataFrame(window_tile_dict)

        if execution_mode == ExecutionMode.LAND_COVER_PREDICTION:

            nodata_rows = (~np.isfinite(window_tile_df)).any(axis=1)

            # Low memory column reindex without copy taken from https://stackoverflow.com/questions/25878198/change-pandas-dataframe-column-order-in-place
            for column in used_columns:
                window_tile_df[column] = window_tile_df.pop(column).replace([np.inf, -np.inf, -np.nan], 0)

            predictions = lc_clf.predict(window_tile_df)

            predictions[nodata_rows] = "nodata"
            predictions = np.reshape(
                predictions, (1, window_kwargs["height"], window_kwargs["width"])
            )
            encoded_predictions = np.zeros_like(predictions, dtype=np.uint8)

            for class_, value in lc_mapping.items():
                encoded_predictions = np.where(
                    predictions == class_, value, encoded_predictions
                )

            with rasterio.open(
                classification_path, "r+", **output_kwargs
            ) as classification_file:
                classification_file.write(encoded_predictions, window=window)

        elif execution_mode == ExecutionMode.SECOND_LEVEL_PREDICTION:

            nodata_rows = (~np.isfinite(window_tile_df)).any(axis=1)

            for column in used_columns:
                window_tile_df[column] = window_tile_df.pop(column).replace([np.inf, -np.inf, -np.nan], 0)

            lc_raster = _get_lc_classification(tile, window)
            lc_raster_flattened = lc_raster.flatten()
            
            sl_predictions = np.empty(len(window_tile_df), dtype=object)

            for lc_class_number , class_model in sl_classifiers.items():
                mask_lc_class = (lc_raster_flattened == lc_class_number)
                
                rows_to_predict = window_tile_df[mask_lc_class]
                if len(rows_to_predict) != 0:
                    sl_predictions[mask_lc_class] = class_model.predict(rows_to_predict)

            sl_predictions[sl_predictions == None] = "noclassified"
            sl_predictions[nodata_rows] = "nodata"
            
            sl_predictions = np.reshape(
                sl_predictions, (1, window_kwargs["height"], window_kwargs["width"])
            )
            encoded_sl_predictions = np.zeros_like(sl_predictions, dtype=np.uint8)

            for class_, value in sl_mapping.items():
                encoded_sl_predictions = np.where(
                    sl_predictions == class_, value, encoded_sl_predictions
                )

            with rasterio.open(
                classification_path, "r+", **output_kwargs
            ) as classification_file:
                classification_file.write(encoded_sl_predictions, window=window)

        print(f"Window {window} processed, output raster {classification_path} updated")

    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{classification_name}",
        file_path=classification_path,
        content_type="image/tif",
    )

    print(f"Classification raster {classification_name} uploaded to Minio")
    print(f"Tile {tile} processed")
    print("Cleaning up temporary files")

    for path in Path(settings.TMP_DIR).glob("**/*"):
        if "classification" in path.name:
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
