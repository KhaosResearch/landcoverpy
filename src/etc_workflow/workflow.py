import collections
import json
from glob import glob
from os.path import join
from pathlib import Path
from shutil import rmtree
from typing import List

import joblib
import numpy as np
import pandas as pd
import rasterio
from distributed import Client

from etc_workflow.aster import get_dem_from_tile
from etc_workflow.config import settings
from etc_workflow.exceptions import EtcWorkflowException, NoSentinelException
from etc_workflow.execution_mode import ExecutionMode
from etc_workflow.utils import (
    _check_tiles_not_predicted_in_training,
    _connect_mongo_composites_collection,
    _connect_mongo_products_collection,
    _create_composite,
    _download_sample_band_by_tile,
    _filter_rasters_paths_by_features_used,
    _get_composite,
    _get_forest_masks,
    _get_kwargs_raster,
    _get_minio,
    _get_product_rasters_paths,
    _get_products_by_tile_and_date,
    _get_raster_filename_from_path,
    _get_raster_name_from_path,
    _group_polygons_by_tile,
    _kmz_to_geojson,
    _mask_polygons_by_tile,
    _read_raster,
    _safe_minio_execute,
    get_season_dict,
    _remove_tiles_already_processed_in_training
)


def workflow(execution_mode: ExecutionMode, client: Client = None, tiles_to_predict: List[str] = None):

    predict = execution_mode != ExecutionMode.TRAINING

    minio = _get_minio()

    # Iterate over a set of geojson/databases (the databases may not be equal)
    geojson_files = []
    for data_class in glob(join(settings.DB_DIR, "*.kmz")):
        if not Path.exists(Path(data_class.replace("kmz","geojson"))):
            print(f"Parsing database to geojson: {data_class}")
            _kmz_to_geojson(data_class)

    for data_class in glob(join(settings.DB_DIR, "*.geojson")):
        print(f"Working with database {data_class}")
        geojson_files.append(data_class)
    polygons_per_tile = _group_polygons_by_tile(*geojson_files)

    if predict:
        print("Predicting tiles")
        if tiles_to_predict is not None:
            polygons_per_tile = {}
            for tile_to_predict in tiles_to_predict:
                polygons_per_tile[tile_to_predict] = []

        tiles = _check_tiles_not_predicted_in_training(list(polygons_per_tile.keys()))

        if execution_mode == ExecutionMode.FOREST_PREDICTION:
            tiles_forest = _check_tiles_not_predicted_in_training(list(polygons_per_tile.keys()), forest_prediction=True)

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

    else:
        print("Creating dataset from tiles")
        # Tiles related to the traininig zone that wasn't already processed
        tiles = _remove_tiles_already_processed_in_training(list(polygons_per_tile.keys()))

        # In training, read all rasters available
        used_columns = None

    if client is not None:
        futures = []
        if execution_mode != ExecutionMode.FOREST_PREDICTION:
            for tile in tiles:
                future = client.submit(_process_tile, tile, execution_mode, polygons_per_tile[tile], used_columns, resources={"Memory": 100})
                futures.append(future)
        else:
            for tile in tiles:
                future = client.submit(_process_tile, tile, ExecutionMode.LAND_COVER_PREDICTION, polygons_per_tile[tile], used_columns, resources={"Memory": 100})
                futures.append(future)
            client.gather(futures, errors="skip")
            for tile in tiles_forest:
                future = client.submit(_process_tile, tile, ExecutionMode.FOREST_PREDICTION, polygons_per_tile[tile], used_columns, resources={"Memory": 100})
                futures.append(future)
        client.gather(futures, errors="skip")

    else:
        if execution_mode != ExecutionMode.FOREST_PREDICTION:
            for tile in tiles:
                try:
                    _process_tile(tile, execution_mode, polygons_per_tile[tile], used_columns)
                except EtcWorkflowException as e:
                    print(e)
        else:
            for tile in tiles:
                try:
                    _process_tile(tile, ExecutionMode.LAND_COVER_PREDICTION, polygons_per_tile[tile], used_columns)
                except EtcWorkflowException as e:
                    print(e)
            for tile in tiles_forest:
                try:
                    _process_tile(tile, ExecutionMode.FOREST_PREDICTION, polygons_per_tile[tile], used_columns)
                except EtcWorkflowException as e:
                    print(e)


    if not predict:
        # Merge all tiles datasets into a big dataset.csv, then upload it to minio

        final_df = None

        minio = _get_minio()

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

            _safe_minio_execute(
                func=minio.fget_object,
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
        _safe_minio_execute(
            func=minio.fput_object,
            bucket_name=settings.MINIO_BUCKET_DATASETS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, "dataset.csv"),
            file_path=file_path,
            content_type="text/csv",
        )


def _process_tile(tile, execution_mode, polygons_in_tile, used_columns=None):

    predict = execution_mode != ExecutionMode.TRAINING

    if not Path(settings.TMP_DIR).exists():
        Path.mkdir(Path(settings.TMP_DIR))

    seasons = get_season_dict()

    minio_client = _get_minio()
    mongo_products_collection = _connect_mongo_products_collection()

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
    max_cloud_percentage = settings.MAX_CLOUD

    spring_start, spring_end = seasons["spring"]
    product_metadata_cursor_spring = _get_products_by_tile_and_date(
        tile, mongo_products_collection, spring_start, spring_end, max_cloud_percentage
    )

    summer_start, summer_end = seasons["summer"]
    product_metadata_cursor_summer = _get_products_by_tile_and_date(
        tile, mongo_products_collection, summer_start, summer_end, max_cloud_percentage
    )

    autumn_start, autumn_end = seasons["autumn"]
    product_metadata_cursor_autumn = _get_products_by_tile_and_date(
        tile, mongo_products_collection, autumn_start, autumn_end, max_cloud_percentage
    )

    product_per_season = {
        "spring": list(product_metadata_cursor_spring)[-settings.MAX_PRODUCTS_COMPOSITE:],
        "autumn": list(product_metadata_cursor_autumn)[-settings.MAX_PRODUCTS_COMPOSITE:],
        "summer": list(product_metadata_cursor_summer)[-settings.MAX_PRODUCTS_COMPOSITE:],
    }

    if (
        len(product_per_season["spring"]) == 0
        or len(product_per_season["autumn"]) == 0
        or len(product_per_season["summer"]) == 0
    ):
        raise NoSentinelException(f"There is no valid Sentinel products for tile {tile}. Skipping it...")

    # Dataframe for storing data of a tile
    tile_df = None

    dems_raster_names = [
        "slope",
        "aspect",
        "dem",
    ]
    
    for dem_name in dems_raster_names:
        # Add dem and aspect data
        if (not predict) or (predict and dem_name in used_columns):

            dem_path = get_dem_from_tile(
                execution_mode, tile, mongo_products_collection, minio_client, dem_name
            )

            # Predict doesnt work yet in some specific tiles where tile is not fully contained in aster rasters
            kwargs = _get_kwargs_raster(dem_path)
            if execution_mode==ExecutionMode.TRAINING:
                crop_mask, label_lon_lat = _mask_polygons_by_tile(polygons_in_tile, kwargs)

            if execution_mode==ExecutionMode.LAND_COVER_PREDICTION:
                crop_mask = np.zeros(shape=(int(kwargs["height"]), int(kwargs["width"])),dtype=np.uint8)

            if execution_mode==ExecutionMode.FOREST_PREDICTION:
                forest_mask = _get_forest_masks(tile)
                crop_mask = np.zeros(shape=(int(kwargs["height"]), int(kwargs["width"])),dtype=np.uint8)

            band_normalize_range = normalize_range.get(dem_name, None)
            raster = _read_raster(
                band_path=dem_path,
                rescale=True,
                normalize_range=band_normalize_range,
            )
            raster_masked = np.ma.masked_array(raster, mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({dem_name: raster_masked})
            tile_df = pd.concat([tile_df, raster_df], axis=1)

    # Get crop mask for sentinel rasters and dataset labeled with database points in tile
    band_path = _download_sample_band_by_tile(tile, minio_client, mongo_products_collection)
    kwargs = _get_kwargs_raster(band_path)

    if execution_mode==ExecutionMode.TRAINING:
        crop_mask, label_lon_lat = _mask_polygons_by_tile(polygons_in_tile, kwargs)

    elif execution_mode==ExecutionMode.LAND_COVER_PREDICTION:
        crop_mask = np.zeros(shape=(int(kwargs["height"]), int(kwargs["width"])), dtype=np.uint8)

    elif execution_mode==ExecutionMode.FOREST_PREDICTION:
        forest_mask = _get_forest_masks(tile)
        crop_mask = np.zeros(shape=(int(kwargs["height"]), int(kwargs["width"])), dtype=np.uint8)

    for season, products_metadata in product_per_season.items():
        print(season)
        bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
        bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES
        current_bucket = None

        if len(products_metadata) == 0:
            raise NoSentinelException(f"There is no valid Sentinel products for tile {tile}. Skipping it...")

        elif len(products_metadata) == 1:
            product_metadata = products_metadata[0]
            current_bucket = bucket_products
        else:
            # If there are multiple products for one season, use a composite.
            mongo_composites_collection = _connect_mongo_composites_collection()
            products_metadata_list = list(products_metadata)
            product_metadata = _get_composite(
                products_metadata_list, mongo_composites_collection, execution_mode
            )
            if product_metadata is None:
                _create_composite(
                    products_metadata_list,
                    minio_client,
                    bucket_products,
                    bucket_composites,
                    mongo_composites_collection,
                    execution_mode
                )
                product_metadata = _get_composite(
                    products_metadata_list, mongo_composites_collection, execution_mode
                )
            current_bucket = bucket_composites

        product_name = product_metadata["title"]

        # For validate dataset geometries, the product name is added.
        if not predict:
            raster_product_name = np.full_like(
                raster_masked, product_name, dtype=object
            )
            raster_df = pd.DataFrame({f"{season}_product_name": raster_product_name})
            tile_df = pd.concat([tile_df, raster_df], axis=1)

        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client, current_bucket
        )

        if predict:
            (rasters_paths, is_band) = _filter_rasters_paths_by_features_used(
                rasters_paths, is_band, used_columns, season
            )
        temp_product_folder = Path(settings.TMP_DIR, product_name + ".SAFE")
        if not temp_product_folder.exists():
            Path.mkdir(temp_product_folder)
        print(f"Processing product {product_name}")

        # Read bands and indexes.
        already_read = []
        for i, raster_path in enumerate(rasters_paths):
            raster_filename = _get_raster_filename_from_path(raster_path)
            raster_name = _get_raster_name_from_path(raster_path)
            temp_path = Path(temp_product_folder, raster_filename)

            # Only keep bands and indexes in indexes_used
            if (not is_band[i]) and (
                not any(
                    raster_name.upper() == index_used.upper()
                    for index_used in indexes_used
                )
            ):
                continue
            # Skip bands in skip_bands
            if is_band[i] and any(
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

            print(f"Downloading raster {raster_name} from minio into {temp_path}")
            _safe_minio_execute(
                func=minio_client.fget_object,
                bucket_name=current_bucket,
                object_name=raster_path,
                file_path=str(temp_path),
            )
            kwargs = _get_kwargs_raster(str(temp_path))
            spatial_resolution = kwargs["transform"][0]
            if spatial_resolution == 10:
                kwargs_10m = kwargs

            band_normalize_range = normalize_range.get(raster_name, None)
            if is_band[i] and (band_normalize_range is None):
                band_normalize_range = (0, 7000)

            raster = _read_raster(
                band_path=temp_path,
                rescale=True,
                normalize_range=band_normalize_range,
            )
            raster_masked = np.ma.masked_array(raster[0], mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked)

            raster_df = pd.DataFrame({f"{season}_{raster_name}": raster_masked})

            tile_df = pd.concat([tile_df, raster_df], axis=1)


    if not predict:

        for index, label in enumerate(["class", "longitude", "latitude", "forest_type"]):

            raster_masked = np.ma.masked_array(label_lon_lat[:, :, index], mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({label: raster_masked})
            tile_df = pd.concat([tile_df, raster_df], axis=1)

        tile_df_name = f"dataset_{tile}.csv"
        tile_df_path = Path(settings.TMP_DIR, tile_df_name)
        tile_df.to_csv(str(tile_df_path), index=False)

        _safe_minio_execute(
            func=minio_client.fput_object,
            bucket_name=settings.MINIO_BUCKET_DATASETS,
            object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/tiles_datasets/{tile_df_name}",
            file_path=tile_df_path,
            content_type="image/tif",
        )

    print("Dataframe information:")
    print(tile_df.info())

    if execution_mode == ExecutionMode.LAND_COVER_PREDICTION:

        model_name = "model.joblib"
        minio_model_folder = settings.LAND_COVER_MODEL_FOLDER
        model_path = join(settings.TMP_DIR, minio_model_folder, model_name)

        _safe_minio_execute(
            func=minio_client.fget_object,
            bucket_name=settings.MINIO_BUCKET_MODELS,
            object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_model_folder}/{model_name}",
            file_path=model_path,
        )

        nodata_rows = (~np.isfinite(tile_df)).any(axis=1)

        # Low memory column reindex without copy taken from https://stackoverflow.com/questions/25878198/change-pandas-dataframe-column-order-in-place
        for column in used_columns:
            tile_df[column] = tile_df.pop(column).replace([np.inf, -np.inf, -np.nan], 0)

        clf = joblib.load(model_path)

        predictions = clf.predict(tile_df)

        predictions[nodata_rows] = "nodata"
        predictions = np.reshape(
            predictions, (1, kwargs_10m["height"], kwargs_10m["width"])
        )
        encoded_predictions = np.zeros_like(predictions, dtype=np.uint8)

        mapping = {
            "nodata": 0,
            "builtUp": 1,
            "herbaceousVegetation": 2,
            "shrubland": 3,
            "water": 4,
            "wetland": 5,
            "cropland": 6,
            "closedForest": 7,
            "openForest": 8,
            "bareSoil": 9
        }
        for class_, value in mapping.items():
            encoded_predictions = np.where(
                predictions == class_, value, encoded_predictions
            )

        kwargs_10m["nodata"] = 0
        kwargs_10m["driver"] = "GTiff"
        kwargs_10m["dtype"] = np.uint8
        classification_name = f"classification_{tile}.tif"
        classification_path = str(Path(settings.TMP_DIR, classification_name))
        with rasterio.open(
            classification_path, "w", **kwargs_10m
        ) as classification_file:
            classification_file.write(encoded_predictions)
        print(f"{classification_name} saved")

        _safe_minio_execute(
            func=minio_client.fput_object,
            bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
            object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{classification_name}",
            file_path=classification_path,
            content_type="image/tif",
        )

    elif execution_mode == ExecutionMode.FOREST_PREDICTION:

        model_name = "model.joblib"
        minio_models_folders_open = settings.OPEN_FOREST_MODEL_FOLDER
        minio_models_folders_dense = settings.DENSE_FOREST_MODEL_FOLDER

        for minio_model_folder in [minio_models_folders_open, minio_models_folders_dense]:

            _safe_minio_execute(
                func=minio_client.fget_object,
                bucket_name=settings.MINIO_BUCKET_MODELS,
                object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_model_folder}/{model_name}",
                file_path=join(settings.TMP_DIR, minio_model_folder, model_name),
            )


        nodata_rows = (~np.isfinite(tile_df)).any(axis=1)

        for column in used_columns:
            tile_df[column] = tile_df.pop(column).replace([np.inf, -np.inf, -np.nan], 0)

        clf_open_forest = joblib.load(join(settings.TMP_DIR, minio_models_folders_open, model_name))
        clf_dense_forest = joblib.load(join(settings.TMP_DIR, minio_models_folders_dense, model_name))

        predictions_open = clf_open_forest.predict(tile_df)
        predictions_dense = clf_dense_forest.predict(tile_df)

        forest_type_vector = forest_mask.flatten()

        predictions = np.where(forest_type_vector==1, "D - " + predictions_dense, "noforest")
        predictions = np.where(forest_type_vector==2, "O - " + predictions_open, predictions)

        predictions[nodata_rows] = "nodata"

        predictions = np.reshape(
            predictions, (1, kwargs_10m["height"], kwargs_10m["width"])
        )
        encoded_predictions = np.zeros_like(predictions, dtype=np.uint8)

        mapping = {
            'nodata': 0,
            'noforest': 1,
            'O - Acebuchales (Olea europaea var. Sylvestris)': 101,
            'O - Encinares (Quercus ilex)': 102,
            'O - Enebrales (Juniperus spp.)': 103,
            'O - Melojares (Quercus pyrenaica)': 104,
            'O - Mezcla de coníferas y frondosas': 105,
            'O - Otras coníferas': 106,
            'O - Otras frondosas': 107,
            'O - Pinar de pino albar (Pinus sylvestris)': 108,
            'O - Pinar de pino carrasco (Pinus halepensis)': 109,
            'O - Pinar de pino negro (Pinus uncinata)': 110,
            'O - Pinar de pino piñonero (Pinus pinea)': 111,
            'O - Pinar de pino salgareño (Pinus nigra)': 112,
            'O - Pinares de pino pinaster': 113,
            'O - Quejigares (Quercus faginea)': 114,
            'O - Robledales de Q. robur y/o Q. petraea': 115,
            'O - Robledales de roble pubescente (Quercus humilis)': 116,
            'O - Sabinares albares (Juniperus thurifera)': 117,
            'O - Sabinares de Juniperus phoenicea': 118,
            'D - Plantacion - Choperas y plataneras de producción': 201,
            'D - Plantacion - Eucaliptales': 202,
            'D - Plantacion - Otras coníferas alóctonas de producción (Larix spp.: Pseudotsuga spp.: etc)': 203,
            'D - Plantacion - Otras especies de producción en mezcla': 204,
            'D - Plantacion - Pinar de pino albar (Pinus sylvestris)': 205,
            'D - Plantacion - Pinar de pino carrasco (Pinus halepensis)': 206,
            'D - Plantacion - Pinar de pino piñonero (Pinus pinea)': 207,
            'D - Plantacion - Pinar de pino radiata': 208,
            'D - Plantacion - Pinar de pino salgareño (Pinus nigra)': 209,
            'D - Plantacion - Pinares de pino pinaster': 210,
            'D - Abedulares (Betula spp.)': 211,
            'D - Abetales (Abies alba)': 212,
            'D - Acebedas (Ilex aquifolium)': 213,
            'D - Acebuchales (Olea europaea var. Sylvestris)': 214,
            'D - Alcornocales (Quercus suber)': 215,
            'D - Avellanedas (Corylus avellana)': 216,
            'D - Bosque ribereño': 217,
            'D - Castañares (Castanea sativa)': 218,
            'D - Encinares (Quercus ilex)': 219,
            'D - Fresnedas (Fraxinus spp.)': 220,
            'D - Hayedos (Fagus sylvatica)': 221,
            'D - Madroñales (Arbutus unedo)': 222,
            'D - Melojares (Quercus pyrenaica)': 223,
            'D - Mezcla de coníferas y frondosas': 224,
            'D - Pinar de pino albar (Pinus sylvestris)': 225,
            'D - Pinar de pino canario (Pinus canariensis)': 226,
            'D - Pinar de pino carrasco (Pinus halepensis)': 227,
            'D - Pinar de pino negro (Pinus uncinata)': 228,
            'D - Pinar de pino piñonero (Pinus pinea)': 229,
            'D - Pinar de pino radiata': 230,
            'D - Pinar de pino salgareño (Pinus nigra)': 231,
            'D - Pinares de pino pinaster': 232,
            'D - Pinsapares (Abies pinsapo)': 233,
            'D - Quejigares (Quercus faginea)': 234,
            'D - Quejigares de Quercus canariensis': 235,
            'D - Robledales de Q. robur y/o Q. petraea': 236,
            'D - Robledales de roble pubescente (Quercus humilis)': 237
        }

        for class_, value in mapping.items():
            encoded_predictions = np.where(
                predictions == class_, value, encoded_predictions
            )

        kwargs_10m["nodata"] = 0
        kwargs_10m["driver"] = "GTiff"
        kwargs_10m["dtype"] = np.uint8
        classification_name = f"forest_classification_{tile}.tif"
        classification_path = str(Path(settings.TMP_DIR, classification_name))
        with rasterio.open(
            classification_path, "w", **kwargs_10m
        ) as classification_file:
            classification_file.write(encoded_predictions)
        print(f"{classification_name} saved")

        _safe_minio_execute(
            func=minio_client.fput_object,
            bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
            object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{classification_name}",
            file_path=classification_path,
            content_type="image/tif",
        )

    for path in Path(settings.TMP_DIR).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
