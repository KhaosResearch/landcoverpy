import datetime
import json
from datetime import datetime
from os.path import join
from typing import List
from pymongo.collection import Collection

import pandas as pd
from sentinelsat.sentinel import SentinelAPI, geojson_to_wkt, read_geojson

from bd_lc_mediterranean.config import settings
from bd_lc_mediterranean.minio import MinioConnection
from bd_lc_mediterranean.mongo import MongoConnection
from bd_lc_mediterranean.utilities.sentinel import _get_sentinel
from bd_lc_mediterranean.utilities.utils import (
    get_products_by_tile_and_date,
    get_season_dict,
)

def _get_available_products(tiles: dict, mongo_collection: Collection, seasons: dict):
    """
    Query to mongo for obtaining products of the given dates and tiles and create a dictionary with all available products in mongo with its cloud percentage.

    Parameters:
        tiles (dict): Dictionary containing tiles names.
        mongo_collection (Collection): Data for mongo access.
        seasons (dict): Dictionary of date range (start and end date) for each query.

    Output:
        tiles (dict): Dictionary order by tile with all available products in mongo and with its cloud percentage.

    """

    # Obtain title and cloud cover percentage of every product available for each tile and each season in the given seasons list
    for tile in tiles:
        for season in seasons:
            start_date, end_date = seasons[season]
            product_metadata_cursor = get_products_by_tile_and_date(
                tile, mongo_collection, start_date, end_date, cloud_percentage=101
            )
            products = list(product_metadata_cursor)

            dict_products = {}
            # Generate a dictionary of the products set with title as key and cloud cover as value
            for product in products:
                title = product["title"]
                dict_products[title] = round(product["indexes"][0]["value"], 6)

            # Add the dictionary of products to the original tiles dictionary by tile and season
            tiles[tile][season] = dict_products

    return tiles


def _get_tiles_in_geojson(sentinel_api: SentinelAPI, geojson_path: str):
    """
    Get set of tiles for a given area defined in a geojson.

    Parameters:
        sentinel_api (SentinelAPI): Sentinel api access data.
        geojson (str): Geojson path to get tiles of. Due to Sentinel-2 limitations, it has to be an small one

    """

    tiles_aoi = {}

    geojson = read_geojson(geojson_path)
    footprint = geojson_to_wkt(geojson)

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 1, 10)

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
        tiles_aoi[tile] = {}

    return tiles_aoi


def compute_cloud_coverage(geojson_path: str):
    """
    Computes the cloud coverage for a list of tiles included in a geojson.

    Parameters:
        countries (List[str]): List of countries included.

    """

    mongo_client = MongoConnection()
    minio_client = MinioConnection()
    mongo_collection = mongo_client.get_collection_object()
    sentinel_api = _get_sentinel()
    tiles = _get_tiles_in_geojson(sentinel_api, geojson_path)

    seasons = get_season_dict()

    tiles_by_country = _get_available_products(
        mongo_collection=mongo_collection, tiles=tiles, seasons=seasons
    )

    tile_metadata_name = "cloud_coverage.json"
    tile_metadata_path = join(settings.TMP_DIR, tile_metadata_name)

    with open(tile_metadata_path, "w") as f:
        json.dump(tiles_by_country, f)

    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_TILE_METADATA,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{tile_metadata_name}",
        file_path=tile_metadata_path,
        content_type="text/json",
    )
