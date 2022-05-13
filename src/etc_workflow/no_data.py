import datetime
import json
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Collection, List
import numpy as np
from minio import Minio
from operator import mul
from itertools import compress

import pandas as pd
from sentinelsat.sentinel import SentinelAPI, geojson_to_wkt, read_geojson

from etc_workflow.config import settings
from etc_workflow.utils import (
    _connect_mongo_products_collection,
    _get_minio,
    _get_products_by_tile_and_date,
    _get_sentinel,
    _safe_minio_execute,
    _get_raster_filename_from_path,
    _get_product_rasters_paths,
    _read_raster,
)


def _get_available_products(tiles: dict, mongo_collection: Collection, seasons: dict):
    """
    Query to mongo for obtaining products of the given dates and tiles and
    create a dictionary with all available products in mongo with its no data percentage.

    Parameters:
        tiles (dict): Dictionary containing tiles names.
        mongo_collection (Collection): Data for mongo access.
        seasons (dict): Dictionary of date range (start and end date) for each query.

    Output:
        tiles (dict): Dictionary order by tile with all available products in mongo and with its no data percentage.

    """

    # Obtain title and no data percentage of every product available for each tile and each season in the given seasons list
    for tile in tiles:
        for season in seasons:
            start_date, end_date = seasons[season]
            product_metadata_cursor = _get_products_by_tile_and_date(
                tile, mongo_collection, start_date, end_date, 101
            )
            products = list(product_metadata_cursor)

            dict_products = {}
            # Generate a dictionary of the products set with title as key and no data percentage as value
            for product in products:
                title = product["title"]
                nodata_percentage = get_nodata_percentage(title, mongo_collection)
                dict_products[title] = nodata_percentage

            # Add the dictionary of products to the original tiles dictionary by tile and season
            tiles[tile][season] = dict_products

    return tiles


def _get_country_tiles(sentinel_api: SentinelAPI, countries: list):
    """
    Get set of tiles for a given area.

    Parameters:
        sentinel_api (SentinelAPI): Sentinel api access data.
        countries (list): Set of countries to get tiles of.

    """

    minio_client = _get_minio()
    tiles_by_country = {}

    for country in countries:
        # Read geojson of the given country
        geojson_filename = str.lower(country) + ".geojson"
        geojson_path = join(settings.TMP_DIR, "countries", geojson_filename)

        _safe_minio_execute(
            func=minio_client.fget_object,
            bucket_name=settings.MINIO_BUCKET_GEOJSONS,
            object_name=join("countries", geojson_filename),
            file_path=geojson_path,
        )

        geojson = read_geojson(geojson_path)
        footprint = geojson_to_wkt(geojson)

        start_date = datetime(2021, 1, 1)
        end_date = datetime(2021, 1, 20)

        products = sentinel_api.query(
            area=footprint,
            filename="S2*",
            producttype="S2MSI2A",
            platformname="Sentinel-2",
            cloudcoverpercentage=(0, 100),
            date=(start_date, end_date),
            order_by="cloudcoverpercentage, +beginposition",
            limit=None,
        )

        products_df = sentinel_api.to_dataframe(products)

        # Get list of unique tiles in the given area.
        tiles = pd.Series(products_df["title"]).str[39:44]
        tiles = pd.unique(tiles)

        # Add tiles to the dictionary
        for tile in tiles:
            tiles_by_country[tile] = {}

    return tiles_by_country


def compute_no_data(countries: List[str], seasons: dict):
    """
    Get percentage of no data for each product in a list of tiles included in several countries.

    Parameters:
        countries (List[str]): List of countries included.
        seasons (dict): Dictionary of date range (start and end date) for each query.

    """

    mongo = _connect_mongo_products_collection()
    minio_client = _get_minio()
    sentinel_api = _get_sentinel()

    tiles = _get_country_tiles(sentinel_api, countries)

    tiles_by_country = _get_available_products(
        mongo_collection=mongo, tiles=tiles, seasons=seasons
    )

    tile_metadata_name = "no_data.json"
    tile_metadata_path = join(settings.TMP_DIR, tile_metadata_name)

    with open(tile_metadata_path, "w") as f:
        json.dump(tiles_by_country, f)

    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_TILE_METADATA,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{tile_metadata_name}",
        file_path=tile_metadata_path,
        content_type="text/json",
    )


def get_nodata_percentage(filename: str, mongo_collection: Collection):
    """
    Read a band and get the no data porcentage in the given product.

    Parameters:
        filename (str): Name of the product to download.
        mongo_collection (Collection): Data for mongo access.

    Returns:
        percentage (float): Percentage of no data in the given product.

    """

    minio_client = _get_minio()
    sample_band = _download_sample_band_by_title(
        filename, minio_client, mongo_collection
    )

    # In case there is no band for the given file (due to errors in the download process it is empty)
    if sample_band == None:
        percentage = None
    else:
        band = _read_raster(sample_band, to_tif=False)

        # Get the number of pixel in the band
        if len(band.shape) == 2:
            n_pixeles = mul(*np.shape(band))
        if len(band.shape) == 3:
            n_pixeles = np.shape(band)[1] * np.shape(band)[2]

        # Calculates percentage of no data
        percentage = (n_pixeles - np.count_nonzero(band)) * 100 / n_pixeles

    return percentage


def _download_sample_band_by_title(
    title: str, minio_client: Minio, mongo_collection: Collection
):
    """
    Having a title of a product, download a sample sentinel band of the product.
    """
    product_metadata = mongo_collection.find_one({"title": title})

    product_path = str(Path(settings.TMP_DIR, title))
    minio_bucket_product = settings.MINIO_BUCKET_NAME_PRODUCTS
    rasters_paths, is_band = _get_product_rasters_paths(
        product_metadata, minio_client, minio_bucket=minio_bucket_product
    )

    if len(is_band) == 0:
        sample_band_path = None
    else:
        sample_band_path_minio = list(compress(rasters_paths, is_band))[4]
        sample_band_path = str(
            Path(product_path, _get_raster_filename_from_path(sample_band_path_minio))
        )
        _safe_minio_execute(
            func=minio_client.fget_object,
            bucket_name=minio_bucket_product,
            object_name=sample_band_path_minio,
            file_path=str(sample_band_path),
        )
    return sample_band_path
