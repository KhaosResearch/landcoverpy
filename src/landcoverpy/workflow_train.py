from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd

from landcoverpy.aster import get_dem_from_tile
from landcoverpy.composite import _create_composite, _get_composite, _validate_composite_products
from landcoverpy.config import settings
from landcoverpy.exceptions import NoSentinelException
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.minio import MinioConnection
from landcoverpy.mongo import MongoConnection
from landcoverpy.utilities.raster import (
    _download_sample_band_by_tile,
    _get_kwargs_raster,
    _get_product_rasters_paths,
    _get_raster_filename_from_path,
    _get_raster_name_from_path,
    _read_raster,
)
from landcoverpy.utilities.utils import (
    _mask_polygons_by_tile,
    get_products_by_tile_and_date,
    get_season_dict,
)

def _process_tile_train(tile, polygons_in_tile, use_aster=True):

    execution_mode = ExecutionMode.TRAINING

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

    print(f"Generating training dataset for tile {tile}")
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
        print(f"Found {len(product_metadata_list)} products for tile {tile} in season {season}")
        product_metadata_list = _validate_composite_products(product_metadata_list)
        print(f"Found {len(product_metadata_list)} valid products for tile {tile} in season {season}")
        # If there are more products than the maximum specified for creating a composite, take the last ones
        product_per_season[season] = product_metadata_list[:settings.MAX_PRODUCTS_COMPOSITE]

        if len(product_per_season[season]) == 0:
            raise NoSentinelException(f"There is no valid Sentinel products for tile {tile} in season {season}. Skipping it...")

    # Dataframe for storing data of a tile
    tile_df = None

    if use_aster:
        dems_raster_names = [
            "slope",
            "aspect",
            "dem",
        ]
    else:
        dems_raster_names = []

    # Get crop mask for sentinel rasters and dataset labeled with database points in tile
    band_path = _download_sample_band_by_tile(tile, minio_client, mongo_products_collection)
    s2_band_kwargs = _get_kwargs_raster(band_path)

    crop_mask, label_lon_lat = _mask_polygons_by_tile(polygons_in_tile, s2_band_kwargs)

    raster = _read_raster(
        band_path=band_path,
        rescale=True
    )

    raster_masked = np.ma.masked_array(raster, mask=crop_mask)
    raster_masked = np.ma.compressed(raster_masked).flatten()
    

    for dem_name in dems_raster_names:
        # Add dem and aspect data
        dem_path = get_dem_from_tile(
            execution_mode, tile, mongo_products_collection, minio_client, dem_name
        )

        band_normalize_range = normalize_range.get(dem_name, None)
        raster = _read_raster(
            band_path=dem_path,
            rescale=True,
            normalize_range=band_normalize_range
        )
        raster_masked = np.ma.masked_array(raster, mask=crop_mask)
        raster_masked = np.ma.compressed(raster_masked).flatten()
        raster_df = pd.DataFrame({dem_name: raster_masked})
        tile_df = pd.concat([tile_df, raster_df], axis=1)

    for season, products_metadata in product_per_season.items():
        print(season)

        if len(products_metadata) == 0:
            raise NoSentinelException(f"There is no valid Sentinel products for tile {tile}. Skipping it...")
        else:
            # If there are multiple products for one season, use a composite.
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


        # For validate dataset geometries, the product name is added.
        raster_product_name = np.full_like(
            raster_masked, product_name, dtype=object
        )
        raster_df = pd.DataFrame({f"{season}_product_name": raster_product_name})
        tile_df = pd.concat([tile_df, raster_df], axis=1)

        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client
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
                bucket_name=minio_bucket,
                object_name=raster_path,
                file_path=str(temp_path),
            )

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

    for index, label in enumerate([settings.LC_PROPERTY, "longitude", "latitude", settings.SL_PROPERTY]):

        raster_masked = np.ma.masked_array(label_lon_lat[:, :, index], mask=crop_mask)
        raster_masked = np.ma.compressed(raster_masked).flatten()
        raster_df = pd.DataFrame({label: raster_masked})
        tile_df = pd.concat([tile_df, raster_df], axis=1)

    tile_df_name = f"dataset_{tile}.csv"
    tile_df_path = Path(settings.TMP_DIR, tile_df_name)
    tile_df.to_csv(str(tile_df_path), index=False)

    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/tiles_datasets/{tile_df_name}",
        file_path=tile_df_path,
        content_type="image/tif",
    )

    print("Dataframe information:")
    print(tile_df.info())

    for path in Path(settings.TMP_DIR).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
