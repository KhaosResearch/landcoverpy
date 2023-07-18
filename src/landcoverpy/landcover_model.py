import json
import random
from pathlib import Path
from shutil import rmtree
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import rasterio

from landcoverpy.aster import get_dem_from_tile
from landcoverpy.config import settings
from landcoverpy.exceptions import NoSentinelException
from landcoverpy.minio import MinioConnection
from landcoverpy.mongo import MongoConnection
from landcoverpy.utilities.geometries import _get_mgrs_from_geometry
from landcoverpy.utilities.raster import (
    _filter_rasters_paths_by_features_used,
    _get_product_rasters_paths,
    _get_raster_filename_from_path,
    _get_raster_name_from_path,
    _read_raster,
)
from landcoverpy.utilities.utils import (
    get_products_by_tile_and_date,
    get_season_dict,
)

class LandcoverModel:

    def __init__(self, model_file, used_columns):
        self.model_file = model_file
        self.used_columns = used_columns

    def predict(self, geojson: str):
        geojson_dict = json.loads(geojson)
        tile = list(_get_mgrs_from_geometry(geojson_dict["features"][0]["geometry"]))[0]
        download_url, prediction_metrics = self._predict_tile(tile, geojson_dict["features"][0]["geometry"])
        return download_url, prediction_metrics

    def _predict_tile(self, tile, geojson_dict):
        if not Path(settings.TMP_DIR).exists():
            Path.mkdir(Path(settings.TMP_DIR))

        seasons = get_season_dict()

        minio_client = MinioConnection()
        mongo_client = MongoConnection()
        mongo_products_collection = mongo_client.get_collection_object()

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
        spring_start, spring_end = seasons["spring"]
        product_metadata_cursor_spring = get_products_by_tile_and_date(
            tile, mongo_products_collection, spring_start, spring_end
        )

        summer_start, summer_end = seasons["summer"]
        product_metadata_cursor_summer = get_products_by_tile_and_date(
            tile, mongo_products_collection, summer_start, summer_end
        )

        autumn_start, autumn_end = seasons["autumn"]
        product_metadata_cursor_autumn = get_products_by_tile_and_date(
            tile, mongo_products_collection, autumn_start, autumn_end
        )

        product_per_season = {
            "spring": min(list(product_metadata_cursor_spring), key=lambda x: x['indexes'][0]['value']),
            "autumn": min(list(product_metadata_cursor_autumn), key=lambda x: x['indexes'][0]['value']),
            "summer": min(list(product_metadata_cursor_summer), key=lambda x: x['indexes'][0]['value']),
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
            if dem_name in self.used_columns:

                dem_path = get_dem_from_tile(
                    tile, mongo_products_collection, minio_client, dem_name
                )

                # Predict doesnt work yet in some specific tiles where tile is not fully contained in aster rasters
                
                band_normalize_range = normalize_range.get(dem_name, None)
                raster, _ = _read_raster(
                    band_path=dem_path,
                    rescale=True,
                    normalize_range=band_normalize_range,
                    mask_geometry=geojson_dict
                )
                raster_df = pd.DataFrame({dem_name: raster.flatten()})
                tile_df = pd.concat([tile_df, raster_df], axis=1)

        for season, product_metadata in product_per_season.items():
            print(season)
            current_bucket = settings.MINIO_BUCKET_NAME_PRODUCTS

            if product_metadata is None:
                raise NoSentinelException(f"There is no valid Sentinel products for tile {tile}. Skipping it...")

            product_name = product_metadata["title"]

            (rasters_paths, is_band) = _get_product_rasters_paths(
                product_metadata, minio_client, current_bucket
            )

            (rasters_paths, is_band) = _filter_rasters_paths_by_features_used(
                rasters_paths, is_band, self.used_columns, season
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
                minio_client.fget_object(
                    bucket_name=current_bucket,
                    object_name=raster_path,
                    file_path=str(temp_path),
                )

                band_normalize_range = normalize_range.get(raster_name, None)
                if is_band[i] and (band_normalize_range is None):
                    band_normalize_range = (0, 7000)

                raster, kwargs_10m  = _read_raster(
                    band_path=temp_path,
                    rescale=True,
                    normalize_range=band_normalize_range,
                    mask_geometry=geojson_dict
                )

                raster_df = pd.DataFrame({f"{season}_{raster_name}": raster.flatten()})

                tile_df = pd.concat([tile_df, raster_df], axis=1)


        print("Dataframe information:")
        print(tile_df.info())

        nodata_rows = (~np.isfinite(tile_df)).any(axis=1)

        # Low memory column reindex without copy taken from https://stackoverflow.com/questions/25878198/change-pandas-dataframe-column-order-in-place
        for column in self.used_columns:
            tile_df[column] = tile_df.pop(column).replace([np.inf, -np.inf, -np.nan], 0)

        clf = joblib.load(self.model_file)

        predictions = clf.predict(tile_df)

        prediction_metrics = {}

        prediction_mean_prob = clf.predict_proba(tile_df)[~nodata_rows].max(axis=1).mean()
        prediction_metrics["prediction_mean_prob"] = prediction_mean_prob

        # When we crop a raster, rows that contains all data to Nan or Inf are those that stay outside the geometry
        out_of_geometry_n_rows = np.sum((~np.isfinite(tile_df)).all(axis=1))
        prediction_metrics["nodata_pixels_percentage"] = (np.sum(nodata_rows) - out_of_geometry_n_rows) / (tile_df.shape[0] - out_of_geometry_n_rows)

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

        random_id = ''.join(random.choices("0123456789", k=8))
        classification_name = f"classification_{random_id}.tif"
        classification_path = str(Path(settings.TMP_DIR, classification_name))
        with rasterio.open(
            classification_path, "w", **kwargs_10m
        ) as classification_file:
            classification_file.write(encoded_predictions)
        print(f"{classification_name} saved")

        minio_client.fput_object(
            bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
            object_name=f"user_requests/{classification_name}",
            file_path=classification_path,
            content_type="image/tif",
        )

        url = minio_client.presigned_get_object(
            bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
            object_name=f"user_requests/{classification_name}",
            expires=timedelta(hours=2),
        )

        for path in Path(settings.TMP_DIR).glob("**/*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)

        print(url)
        return url, prediction_metrics
