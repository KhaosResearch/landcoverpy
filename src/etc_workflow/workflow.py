import json
from datetime import datetime
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
from etc_workflow.utils import (
    _check_tiles_unpredicted_in_training,
    _connect_mongo_composites_collection,
    _connect_mongo_products_collection,
    _create_composite,
    _download_sample_band_by_tile,
    _filter_rasters_paths_by_features_used,
    _get_composite,
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


def workflow(predict: bool, client: Client = None, tiles_to_predict: List[str] = None):

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

        tiles = _check_tiles_unpredicted_in_training(list(polygons_per_tile.keys()))

        # For predictions, read the rasters used in "metadata.json".
        metadata_filename = "metadata.json"
        metadata_filepath = join(settings.TMP_DIR, metadata_filename)

        _safe_minio_execute(
            func=minio.fget_object,
            bucket_name=settings.MINIO_BUCKET_MODELS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, metadata_filename),
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
        for tile in tiles:
            future = client.submit(
                _process_tile,
                tile,
                predict,
                polygons_per_tile[tile],
                used_columns,
                resources={"Memory": 100},
            )
            futures.append(future)
        client.gather(futures)
    else:
        for tile in tiles:
            _process_tile(tile, predict, polygons_per_tile[tile], used_columns)

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


def _process_tile(tile, predict, polygons_in_tile, used_columns=None):

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
    max_cloud_percentage = settings.MAX_CLOUD_PERCENTAGE

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
        "spring": list(product_metadata_cursor_spring)[:5],
        "autumn": list(product_metadata_cursor_autumn)[:5],
        "summer": list(product_metadata_cursor_summer)[:5],
    }

    if (
        len(product_per_season["spring"]) == 0
        or len(product_per_season["autumn"]) == 0
        or len(product_per_season["summer"]) == 0
    ):
        print(f"There is no valid data for tile {tile}. Skipping it...")
        return

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
                tile, mongo_products_collection, minio_client, dem_name
            )

            # Predict doesnt work yet in some specific tiles where tile is not fully contained in aster rasters
            kwargs = _get_kwargs_raster(dem_path)
            if not predict:
                crop_mask, label_lon_lat = _mask_polygons_by_tile(
                    polygons_in_tile, kwargs
                )
            if predict:
                crop_mask = np.zeros(
                    shape=(int(kwargs["height"]), int(kwargs["width"]))
                )

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

    if not predict:
        crop_mask, label_lon_lat = _mask_polygons_by_tile(polygons_in_tile, kwargs)
    if predict:
        crop_mask = np.zeros(shape=(int(kwargs["height"]), int(kwargs["width"])))

    for season, products_metadata in product_per_season.items():
        print(season)
        bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
        bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES
        current_bucket = None

        if len(products_metadata) == 0:
            print(f"Products found in tile {tile} are not valid")
            break

        elif len(products_metadata) == 1:
            product_metadata = products_metadata[0]
            current_bucket = bucket_products
        else:
            # If there are multiple products for one season, use a composite.
            mongo_composites_collection = _connect_mongo_composites_collection()
            products_metadata_list = list(products_metadata)
            product_metadata = _get_composite(
                products_metadata_list, mongo_composites_collection
            )
            if product_metadata is None:
                _create_composite(
                    products_metadata_list,
                    minio_client,
                    bucket_products,
                    bucket_composites,
                    mongo_composites_collection,
                )
                product_metadata = _get_composite(
                    products_metadata_list, mongo_composites_collection
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

        raster_masked = np.ma.masked_array(label_lon_lat[:, :, 0], mask=crop_mask)
        raster_masked = np.ma.compressed(raster_masked).flatten()
        raster_df = pd.DataFrame({"class": raster_masked})
        tile_df = pd.concat([tile_df, raster_df], axis=1)

        raster_masked = np.ma.masked_array(label_lon_lat[:, :, 1], mask=crop_mask)
        raster_masked = np.ma.compressed(raster_masked).flatten()
        raster_df = pd.DataFrame({"longitude": raster_masked})
        tile_df = pd.concat([tile_df, raster_df], axis=1)

        raster_masked = np.ma.masked_array(label_lon_lat[:, :, 2], mask=crop_mask)
        raster_masked = np.ma.compressed(raster_masked).flatten()
        raster_df = pd.DataFrame({"latitude": raster_masked})
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

    if predict:

        model_name = "model.joblib"
        model_path = join(settings.TMP_DIR, model_name)

        _safe_minio_execute(
            func=minio_client.fget_object,
            bucket_name=settings.MINIO_BUCKET_MODELS,
            object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{model_name}",
            file_path=model_path,
        )

        kwargs_10m["nodata"] = 0
        clf = joblib.load(model_path)
        predict_df = tile_df
        predict_df.sort_index(inplace=True, axis=1)
        predict_df = predict_df.replace([np.inf, -np.inf], np.nan)
        nodata_rows = np.isnan(predict_df).any(axis=1)
        predict_df.fillna(0, inplace=True)
        predict_df = predict_df.reindex(columns=used_columns)
        predictions = clf.predict(predict_df)
        predictions[nodata_rows] = "nodata"
        predictions = np.reshape(
            predictions, (1, kwargs_10m["height"], kwargs_10m["width"])
        )
        encoded_predictions = predictions.copy()
        
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
                encoded_predictions == class_, value, encoded_predictions
            )

        encoded_predictions = encoded_predictions.astype(np.uint8)

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

    for path in Path(settings.TMP_DIR).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
