import json
from os.path import join
from typing import Collection, List
import numpy as np
from operator import mul

from etc_workflow.config import settings
from etc_workflow.utils import (
    _connect_mongo_products_collection,
    _get_minio,
    _get_products_by_tile_and_date,
    _safe_minio_execute,
    _read_raster,
    _download_sample_band_by_title,
    get_season_dict
)


def _get_dict_of_products_by_tile(tiles: List[str], mongo_collection: Collection, seasons: dict):
    """
    Query to mongo for obtaining products of the given seasons and tiles.
    Then, creates a dictionary with all available products related with its no data percentage.

    Parameters:
        tiles (List[str]): List of tiles names.
        mongo_collection (Collection): Data for mongo access.
        seasons (dict): Dictionary which keys are the name of the seasons. The values are tuples containing the start date and the ending date of each season

    Output:
        tiles (dict): Dictionary order by tile with all available products in mongo and with its no data percentage.

    """

    products_by_tiles = {}
    # Obtain title and no data percentage of every product available for each tile and each season in the given seasons list
    for tile in tiles:
        products_by_tiles[tile] = {}
        for season in seasons:
            start_date, end_date = seasons[season]
            product_metadata_cursor = _get_products_by_tile_and_date(
                tile, mongo_collection, start_date, end_date, cloud_percentage=101
            )
            products = list(product_metadata_cursor)

            dict_products = {}
            # Generate a dictionary of the products set with title as key and no data percentage as value
            for product in products:
                title = product["title"]
                nodata_percentage = _get_nodata_percentage(title, mongo_collection)
                dict_products[title] = nodata_percentage

            # Add the dictionary of products to the output dictionary by tile and season
            products_by_tiles[tile][season] = dict_products

    return products_by_tiles

def _get_nodata_percentage(product_title: str, mongo_collection: Collection):
    """
    Read a 10m sample band of a product and get the no data porcentage in the given product.

    Parameters:
        product_title (str): Name of the product to download.
        mongo_collection (Collection): Mongo collection.

    Returns:
        percentage (float): Percentage of no data in the given product.

    """

    minio_client = _get_minio()
    sample_band = _download_sample_band_by_title(
        product_title, minio_client, mongo_collection
    )

    band = _read_raster(sample_band, to_tif=False)

    # Get the number of pixel in the band
    if len(band.shape) == 2:
        n_pixeles = mul(*np.shape(band))
    if len(band.shape) == 3:
        n_pixeles = np.shape(band)[1] * np.shape(band)[2]

    # Calculates percentage of no data
    percentage = (n_pixeles - np.count_nonzero(band)) * 100 / n_pixeles

    return percentage

def get_quality_map_nodata(tiles: List[str]):
    """
    Get percentage of no data for each product in a list of tiles.

    Parameters:
        tiles (List[str]): List of tiles used to compute the quality map
    """

    mongo = _connect_mongo_products_collection()
    minio_client = _get_minio()

    products_by_tiles = _get_dict_of_products_by_tile(
        tiles=tiles, mongo_collection=mongo, seasons=get_season_dict()
    )

    tile_metadata_name = "no_data.json"
    tile_metadata_path = join(settings.TMP_DIR, tile_metadata_name)

    with open(tile_metadata_path, "w") as f:
        json.dump(products_by_tiles, f)

    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_TILE_METADATA,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{tile_metadata_name}",
        file_path=tile_metadata_path,
        content_type="text/json",
    )