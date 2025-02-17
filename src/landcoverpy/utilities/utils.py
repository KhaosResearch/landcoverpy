import json
from datetime import datetime
from itertools import compress
from os.path import join
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyproj
import rasterio
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from shapely.geometry import Point
from shapely.ops import transform
from sklearn.decomposition import PCA
from rasterio.windows import Window

from landcoverpy.config import settings
from landcoverpy.minio import MinioConnection
from landcoverpy.utilities.raster import (
    _get_product_rasters_paths,
    _get_raster_filename_from_path,
    _read_raster,
)


def get_season_dict():

    with open(settings.SEASONS_FILE) as f:
        seasons = json.load(f)

    seasons_out = {}
    
    for season in seasons:
        seasons_out[season] = (datetime.strptime(seasons[season]["start"], '%Y-%m-%d'), datetime.strptime(seasons[season]["end"], '%Y-%m-%d'))

    return seasons_out

def get_products_by_tile_and_date(
    tile: str,
    mongo_collection: Collection,
    start_date: datetime,
    end_date: datetime,
    min_useful_data_percentage,
) -> Cursor:
    """
    Query to mongo for obtaining products filtered by tile, date and min_useful_data_percentage. It does not work with composites.
    """

    pipeline = [
        {
            "$match": {
                "tile": tile,
                "datetakeSensingTime": {"$gte": start_date, "$lt": end_date}
            }
        },
        {
            "$addFields": {
                "usefulPixelsPercentage": {
                    "$subtract": [
                        100,
                        {
                            "$add": [
                                {"$multiply": ["$intermediateProducts.cloudmask.rasterMeanValue", 100]},
                                "$noDataPercentage"
                            ]
                        }
                    ]
                }
            }
        },
        {
            "$match": {
                "$expr": {"$gt": ["$usefulPixelsPercentage", min_useful_data_percentage]}
            }
        },
        {
            "$sort": {"usefulPixelsPercentage": -1}
        }
    ]

    product_metadata_cursor = mongo_collection.aggregate(pipeline)

    return product_metadata_cursor

def _pca(data: pd.DataFrame, variance_explained: int = 75):
    """
    Return the main columns after a Principal Component Analysis.

    Source:
        https://bitbucket.org/khaosresearchgroup/enbic2lab-images/src/master/soil/PCA_variance/pca-variance.py
    """

    def _dimension_reduction(percentage_variance: list, variance_explained: int):
        cumulative_variance = 0
        n_components = 0
        while cumulative_variance < variance_explained:
            cumulative_variance += percentage_variance[n_components]
            n_components += 1
        return n_components

    pca = PCA()
    pca_data = pca.fit_transform(data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    number_components = _dimension_reduction(per_var, variance_explained)

    pca = PCA(number_components)
    pca_data = pca.fit_transform(data)
    pc_labels = ["PC" + str(x) for x in range(1, number_components + 1)]
    pca_df = pd.DataFrame(data=pca_data, columns=pc_labels)

    pca_df.to_csv("PCA_plot.csv", sep=",", index=False)

    col = []
    for columns in np.arange(number_components):
        col.append("PC" + str(columns + 1))

    loadings = pd.DataFrame(pca.components_.T, columns=col, index=data.columns)
    loadings.to_csv("covariance_matrix.csv", sep=",", index=True)

    # Extract the most important column from each PC https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
    col_ids = [np.abs(pc).argmax() for pc in pca.components_]
    column_names = data.columns
    col_names_list = [column_names[id_] for id_ in col_ids]
    col_names_list = list(dict.fromkeys(col_names_list))
    col_names_list.sort()
    return col_names_list

def _label_neighbours(
    height: int,
    width: int,
    row: int,
    column: int,
    coordinates: Tuple[int, int],
    label: str,
    forest_type: str,
    label_lon_lat: np.ndarray,
) -> np.ndarray:
    """
    Label an input dataset in an area of 3x3 pixels being the center the position (row, column).

    Parameters:
        height (int) : original raster height.
        width (int) : original raster width.
        row (int) : dataset row position to label.
        column (int) : dataset column position to label.
        coordinates (Tuple[int, int]) : latitude and longitude of the point
        label (str) : label name.
        forest_type (str) : Type of the forest, None is pixel is not related to forests
        label_lon_lat (np.ndarray) : empty array of size (height, width, 3)

    Returns:
        label_lon_lat (np.ndarray) : 3D array containing the label and coordinates (lat and lon) of the point.
                                     It can be indexed as label_lon_lat[pixel_row, pixel_col, i].
                                     `i` = 0 refers to the label of the pixel,
                                     `i` = 1 refers to the longitude of the pixel,
                                     `i` = 2 refers to the latitude of the pixel,
                                     `i` = 3 refers to the forest type of the pixel,
    """
    # check the pixel is not out of bounds
    top = 0 < row + 1 < height
    bottom = 0 < row - 1 < height
    left = 0 < column - 1 < width
    right = 0 < column + 1 < width

    label_lon_lat[row, column, :] = label, coordinates[0], coordinates[1], forest_type

    if top:
        label_lon_lat[row - 1, column, :] = label, coordinates[0], coordinates[1], forest_type

        if right:
            label_lon_lat[row, column + 1, :] = label, coordinates[0], coordinates[1], forest_type
            label_lon_lat[row - 1, column + 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
                forest_type
            )

        if left:
            label_lon_lat[row - 1, column - 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
                forest_type
            )
            label_lon_lat[row, column - 1, :] = label, coordinates[0], coordinates[1], forest_type

    if bottom:
        label_lon_lat[row + 1, column, :] = label, coordinates[0], coordinates[1], forest_type

        if left:
            label_lon_lat[row + 1, column - 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
                forest_type
            )
            label_lon_lat[row, column - 1, :] = label, coordinates[0], coordinates[1], forest_type

        if right:
            label_lon_lat[row, column + 1, :] = label, coordinates[0], coordinates[1], forest_type
            label_lon_lat[row + 1, column + 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
                forest_type
            )

    return label_lon_lat

def _mask_polygons_by_tile(
    polygons_in_tile: dict, kwargs: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """'
    Label all the pixels in a dataset from points databases for a given tile.

    Parameters:
        polygons_in_tile (dict) : Dictionary of points to label.
        kwargs (dict) : Metadata of the raster used.

    Returns:
        band_mask (np.ndarray) : Mask for the `label_lon_lat matrix`, indicates if a pixel is labeled or not.
        label_lon_lat (np.ndarray) : 3D array containing the label and coordinates (lat and lon) of the point.
                                     It can be indexed as label_lon_lat[pixel_row, pixel_col, i].
                                     `i` = 0 refers to the label of the pixel,
                                     `i` = 1 refers to the longitude of the pixel,
                                     `i` = 2 refers to the latitude of the pixel,

    """
    label_lon_lat = np.zeros((kwargs["height"], kwargs["width"], 4), dtype=object)

    # Label all the pixels in points database
    for geometry_id in range(len(polygons_in_tile)):
        # Get point and label
        geometry_raw = polygons_in_tile[geometry_id]["geometry"]["coordinates"]
        geometry = Point(geometry_raw[0], geometry_raw[1])
        label = polygons_in_tile[geometry_id]["properties"][settings.LC_PROPERTY]

        # Transform point projection to original raster pojection
        project = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), kwargs["crs"], always_xy=True
        ).transform
        tr_point = transform(project, geometry)

        # Get matrix position from the pixel corresponding to a given point coordinates
        row, column = rasterio.transform.rowcol(
            kwargs["transform"], tr_point.x, tr_point.y
        )
        sl_label = polygons_in_tile[geometry_id]["properties"].get(settings.SL_PROPERTY, None)

        label_lon_lat = _label_neighbours(
            kwargs["height"],
            kwargs["width"],
            row,
            column,
            geometry_raw,
            label,
            sl_label,
            label_lon_lat,
        )

    # Get mask from labeled dataset.
    band_mask = label_lon_lat[:, :, 0] == 0

    return band_mask, label_lon_lat

def _check_tiles_not_predicted(tiles: List[str], second_level_prediction: bool = False):

    minio = MinioConnection()

    if second_level_prediction:
        prefix = join(settings.MINIO_DATA_FOLDER_NAME, "sl_classification")
    else:
        prefix = join(settings.MINIO_DATA_FOLDER_NAME, "classification")

    classification_raster_cursor = minio.list_objects(
        settings.MINIO_BUCKET_CLASSIFICATIONS,
        prefix=prefix
    )

    predicted_tiles = []
    for classification_raster in classification_raster_cursor:
        classification_raster_cursor_path = classification_raster.object_name
        predicted_tile = classification_raster_cursor_path[
            -9:-4
        ]  # ...classification_99XXX.tif
        predicted_tiles.append(predicted_tile)

    unpredicted_tiles = list(np.setdiff1d(tiles, predicted_tiles))

    return unpredicted_tiles

def _get_lc_classification(tile: str, window: Window = None):
    """
    Get the land cover classification from a certain tile.
    Optionally, a window can be passed to read only a portion of the raster.
    """

    minio_client = MinioConnection()
    filename = f"classification_{tile}.tif"
    band_path = join(settings.TMP_DIR, filename)

    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, filename),
        file_path=band_path,
    )

    with rasterio.open(band_path) as band_file:
        band = band_file.read(window=window)

    return band

def _remove_tiles_already_processed_in_training(tiles_in_training: List[str]):

    minio = MinioConnection()

    tiles_datasets_cursor = minio.list_objects(
        settings.MINIO_BUCKET_DATASETS,
        prefix=join(settings.MINIO_DATA_FOLDER_NAME, "tiles_datasets", ""),
    )

    tiles_processed = []
    for tile_dataset in tiles_datasets_cursor:
        tile_dataset_cursor_path = tile_dataset.object_name
        tile_processed = tile_dataset_cursor_path[
            -9:-4
        ]  # ...dataset_99XXX.csv
        tiles_processed.append(tile_processed)

    unprocessed_tiles = list(np.setdiff1d(tiles_in_training, tiles_processed))

    return unprocessed_tiles