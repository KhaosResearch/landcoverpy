import json
import traceback
import warnings
from datetime import datetime
from functools import partial
from hashlib import sha256
from itertools import compress
from logging import exception
from os.path import join
from pathlib import Path
from typing import Callable, Iterable, List, Tuple
from zipfile import ZipFile
from bs4 import BeautifulSoup 
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from mgrs import MGRS
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from rasterio import mask as msk
from rasterio.warp import Resampling, reproject
from scipy.ndimage import convolve
from sentinelsat.sentinel import  read_geojson
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import transform
from sklearn.decomposition import PCA

from etc_workflow import raw_index_calculation_composite
from etc_workflow.config import settings
from etc_workflow.execution_mode import ExecutionMode
from etc_workflow.rasterpoint import RasterPoint
from etc_workflow.minio_connection import MinioConnection



def get_season_dict():
    spring_start = datetime.strptime(settings.SPRING_START, '%Y-%m-%d')
    spring_end = datetime.strptime(settings.SPRING_END, '%Y-%m-%d')
    summer_start = datetime.strptime(settings.SUMMER_START, '%Y-%m-%d')
    summer_end = datetime.strptime(settings.SUMMER_END, '%Y-%m-%d')
    autumn_start = datetime.strptime(settings.AUTUMN_START, '%Y-%m-%d')
    autumn_end = datetime.strptime(settings.AUTUMN_END, '%Y-%m-%d')
    
    seasons =   {
        "spring" : (spring_start, spring_end),
        "summer" : (summer_start, summer_end),
        "autumn" : (autumn_start, autumn_end)
    }

    return seasons




def _group_polygons_by_tile(*geojson_files: str) -> dict:
    """
    Extracts coordinates of geometries from specific geojson files, then creates a mapping [Sentinel's tile -> List of geometries contained in that tile].
    """
    tiles = {}

    for geojson_file in geojson_files:
        geojson = read_geojson(geojson_file)

        print(f"Querying relevant tiles for {len(geojson['features'])} features")
        for feature in geojson["features"]:
            small_geojson = {"type": "FeatureCollection", "features": [feature]}
            geometry = small_geojson["features"][0]["geometry"]
            properties = small_geojson["features"][0]["properties"]
            classification_label = geojson_file.split("_")[2]
            intersection_tiles = _get_mgrs_from_geometry(geometry)

            for tile in intersection_tiles:
                if tile not in tiles:
                    tiles[tile] = []

                tiles[tile].append(
                    {
                        "label": classification_label,
                        "geometry": geometry,
                        "properties": properties
                    }
                )
    return tiles


def _get_mgrs_from_geometry(geometry: dict):
    """
    Get the MGRS coordinates for the 4 corners of the bounding box of a geometry

    This wont work for geometry bigger than a tile. A chunk of the image could not fit between the products of the 4 corners
    """
    tiles = set()
    corners = _get_corners_geometry(geometry)
    for point in corners.values():
        tiles.add(MGRS().toMGRS(*point, MGRSPrecision=0))

    return tiles


def _get_corners_geometry(geometry: dict):
    """
    Get the coordinates of the 4 corners of the bounding box of a geometry
    """

    coordinates = geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        coordinates = coordinates[0]  # TODO multiple polygons in a geometry
    lon = []
    lat = []
    if geometry["type"] == "Point":
        lon.append(coordinates[0])
        lat.append(coordinates[1])
    else:
        coordinates = coordinates[
            0
        ]  # Takes only the outer ring of the polygon: https://geojson.org/geojson-spec.html#polygon
        for coordinate in coordinates:
            lon.append(coordinate[0])
            lat.append(coordinate[1])

    max_lon = max(lon)
    min_lon = min(lon)
    max_lat = max(lat)
    min_lat = min(lat)

    return {
        "top_left": (max_lat, min_lon),
        "top_right": (max_lat, max_lon),
        "bottom_left": (min_lat, min_lon),
        "bottom_right": (min_lat, max_lon),
    }


def get_products_by_tile_and_date(
    tile: str,
    mongo_collection: Collection,
    start_date: datetime,
    end_date: datetime,
    cloud_percentage,
) -> Cursor:
    """
    Query to mongo for obtaining products filtered by tile, date and cloud percentage
    """
    product_metadata_cursor = mongo_collection.aggregate(
        [
            {
                "$project": {
                    "_id": 1,
                    "indexes": {
                        "$filter": {
                            "input": "$indexes",
                            "as": "index",
                            "cond": {
                                "$and": [
                                    {"$eq": ["$$index.mask", None]},
                                    {"$eq": ["$$index.name", "cloud-mask"]},
                                    {"$lt": ["$$index.value", cloud_percentage]},
                                ]
                            },
                        }
                    },
                    "id": 1,
                    "title": 1,
                    "size": 1,
                    "date": 1,
                    "creationDate": 1,
                    "ingestionDate": 1,
                    "objectName": 1,
                }
            },
            {
                "$match": {
                    "indexes.0": {"$exists": True},
                    "title": {"$regex": f"_T{tile}_"},
                    "date": {
                        "$gte": start_date,
                        "$lte": end_date,
                    },
                }
            },
        ]
    )

    return product_metadata_cursor

def _get_centroid(geometry: dict):
    """
    Only works for non-self-intersecting closed polygon. The vertices are assumed
    to be numbered in order of their occurrence along the polygon's perimeter;
    furthermore, the vertex ( xn, yn ) is assumed to be the same as ( x0, y0 ),
    meaning i + 1 on the last case must loop around to i = 0.

    Source: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    """
    coordinates = geometry["coordinates"][
        0
    ]  # Takes only the outer ring of the polygon: https://geojson.org/geojson-spec.html#polygon
    lon = []
    lat = []
    for coordinate in coordinates:
        lon.append(coordinate[0])
        lat.append(coordinate[1])

    # Calculate sums
    sum_lon = 0
    sum_lat = 0
    sum_A = 0
    for i in range(len(lon) - 1):
        sum_lon += (lon[i] + lon[i + 1]) * (
            (lon[i] * lat[i + 1]) - (lon[i + 1] * lat[i])
        )
        sum_lat += (lat[i] + lat[i + 1]) * (
            (lon[i] * lat[i + 1]) - (lon[i + 1] * lat[i])
        )
        sum_A += (lon[i] * lat[i + 1]) - (lon[i + 1] * lat[i])

    # Calculate area inside the polygon's signed area
    A = sum_A / 2

    # Calculate centroid coordinates
    Clon = (1 / (6 * A)) * sum_lon
    Clat = (1 / (6 * A)) * sum_lat

    return (Clat, Clon)


def _filter_valid_products(products_metadata: List, minio_client: MinioConnection):
    """
    Check a list of products and filter those that are not valid.
    A product is not valid if it has more than a 20% of nodata pixels.
    """
    is_valid = []
    bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
    for product_metadata in products_metadata:
        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client, bucket_products
        )
        rasters_paths = list(compress(rasters_paths, is_band))
        sample_band_path = rasters_paths[0]
        sample_band_filename = _get_raster_filename_from_path(sample_band_path)
        file_path = str(
            Path(settings.TMP_DIR, product_metadata["title"], sample_band_filename)
        )
        minio_client.fget_object(bucket_products, sample_band_path, file_path)
        sample_band = _read_raster(file_path)
        num_pixels = np.product(sample_band.shape)
        num_nans = np.isnan(sample_band).sum()
        nan_percentage = num_nans / num_pixels
        is_valid.append(nan_percentage < 0.2)
    products_metadata = list(compress(products_metadata, is_valid))
    return products_metadata


def _get_product_rasters_paths(
    product_metadata: dict, minio_client: MinioConnection, minio_bucket: str
) -> Tuple[Iterable[str], Iterable[bool]]:
    """
    Get the paths to all rasters of a product in minio.
    Another list of boolean is returned, it points out if each raster is a sentinel band or not (e.g. index).
    """
    product_title = product_metadata["title"]
    product_dir = None

    if minio_bucket == settings.MINIO_BUCKET_NAME_PRODUCTS:
        year = product_metadata["date"].strftime("%Y")
        month = product_metadata["date"].strftime("%B")
        product_dir = f"{year}/{month}/{product_title}"

    elif minio_bucket == settings.MINIO_BUCKET_NAME_COMPOSITES:
        product_dir = product_title

    bands_dir = f"{product_dir}/raw/"
    indexes_dir = f"{product_dir}/indexes/{product_title}/"

    bands_paths = minio_client.list_objects(minio_bucket, prefix=bands_dir)
    indexes_path = minio_client.list_objects(minio_bucket, prefix=indexes_dir)

    rasters = []
    is_band = []
    for index_path in indexes_path:
        rasters.append(index_path.object_name)
        is_band.append(False)
    for band_path in bands_paths:
        rasters.append(band_path.object_name)
        is_band.append(True)

    return (rasters, is_band)


def _project_shape(geom: dict, scs: str = "epsg:4326", dcs: str = "epsg:32630"):
    """
    Project a shape from a source coordinate system to another one.

    Parameters:
        geom (dict) : Input geometry.
        scs (str) : Source coordinate system.
        dcs (str) : Destination coordinate system.

    Returns:
        p_geom (dict) : Geometry proyected to destination coordinate system
    """
    # TODO remove this warning catcher
    # This disables FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6 in_crs_string = _prepare_from_proj_string(in_crs_string)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        project = partial(
            pyproj.transform, pyproj.Proj(init=scs), pyproj.Proj(init=dcs)
        )

    return transform(project, shape(geom))



def _normalize(matrix, value1, value2):
    """
    Normalize a numpy matrix with linear function using a range for the normalization.
    Parameters:
        matrix (np.ndarray) : Matrix to normalize.
        value1 (float) : Value mapped to -1 in normalization.
        value2 (float) : Value mapped to 1 in normalization.

    Returns:
        normalized_matrix (np.ndarray) : Matrix normalized.

    """
    try:
        matrix = matrix.astype(dtype=np.float32)
        matrix[matrix == -np.inf] = np.nan
        matrix[matrix == np.inf] = np.nan
        # calculate linear function
        m = 2.0 / (value2 - value1)
        n = 1.0 - m * value2
        normalized_matrix = m * matrix + n
    except exception:
        normalized_matrix = matrix
    return normalized_matrix


def _download_sample_band_by_tile(tile: str, minio_client: MinioConnection, mongo_collection: Collection):
    """
    Having a tile, download a 10m sample sentinel band of any related product.
    """
    product_metadata = mongo_collection.find_one({"title": {"$regex": f"_T{tile}_"}})
    product_title = product_metadata["title"]
    sample_band_path = _download_sample_band_by_title(product_title, minio_client, mongo_collection)
    return sample_band_path

def _download_sample_band_by_title(
    title: str, minio_client: MinioConnection, mongo_collection: Collection
):
    """
    Having a title of a product, download a 10m sample sentinel band of the product.
    """
    product_metadata = mongo_collection.find_one({"title": title})

    product_path = str(Path(settings.TMP_DIR, title))
    minio_bucket_product = settings.MINIO_BUCKET_NAME_PRODUCTS
    rasters_paths, is_band = _get_product_rasters_paths(
        product_metadata, minio_client, minio_bucket=minio_bucket_product
    )
    sample_band_paths_minio = list(compress(rasters_paths, is_band))
    for sample_band_path_minio in sample_band_paths_minio:
        sample_band_path = str(
            Path(product_path, _get_raster_filename_from_path(sample_band_path_minio))
        )
        minio_client.fget_object(minio_bucket_product, sample_band_path_minio, str(sample_band_path))
        if _get_spatial_resolution_raster(sample_band_path) == 10:
            return sample_band_path

    # If no bands of 10m is available in minio
    raise RuntimeError(f"Either data of product {title} wasn't found in MinIO or 10m bands weren't found for that product.")


def _get_raster_filename_from_path(raster_path):
    """
    Get a filename from a raster's path
    """
    return raster_path.split("/")[-1]


def _get_raster_name_from_path(raster_path):
    """
    Get a raster's name from a raster's path
    """
    raster_filename = _get_raster_filename_from_path(raster_path)
    return raster_filename.split(".")[0].split("_")[0]


def _get_spatial_resolution_raster(raster_path):
    """
    Get a raster's spatial resolution from a raster's path
    """
    kwargs = _get_kwargs_raster(raster_path)
    return kwargs["transform"][0]


def _get_kwargs_raster(raster_path):
    """
    Get a raster's metadata from a raster's path
    """
    with rasterio.open(raster_path) as raster_file:
        kwargs = raster_file.meta
        return kwargs


def _sentinel_date_to_datetime(date: str):
    """
    Parse a string date (YYYYMMDDTHHMMSS) to a sentinel datetime
    """
    date_datetime = datetime(
        int(date[0:4]),
        int(date[4:6]),
        int(date[6:8]),
        int(date[9:11]),
        int(date[11:13]),
        int(date[13:15]),
    )
    return date_datetime


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


def _convert_3D_2D(geometry):
    """
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    """
    out_geo = geometry
    if geometry.has_z:
        if geometry.geom_type == "Polygon":
            lines = [xy[:2] for xy in list(geometry.exterior.coords)]
            new_p = Polygon(lines)
            out_geo = new_p
        elif geometry.geom_type == "MultiPolygon":
            new_multi_p = []
            for ap in geometry.geoms:
                lines = [xy[:2] for xy in list(ap.exterior.coords)]
                new_p = Polygon(lines)
                new_multi_p.append(new_p)
            out_geo = MultiPolygon(new_multi_p)
    return out_geo


def _filter_rasters_paths_by_features_used(
    rasters_paths: List[str], is_band: List[bool], used_columns: List[str], season: str
) -> Tuple[Iterable[str], Iterable[bool]]:
    """
    Filter a list of rasters paths by a list of raster names (obtained in feature reduction).
    """
    pc_raster_paths = []
    season_used_columns = []
    already_read = []
    is_band_pca = []
    for pc_column in used_columns:
        if season in pc_column:
            season_used_columns.append(pc_column.split("_")[-1])
    for i, raster_path in enumerate(rasters_paths):
        raster_name = _get_raster_name_from_path(raster_path)
        raster_name = raster_name.split("_")[-1]
        if any(x == raster_name for x in season_used_columns) and (
            raster_name not in already_read
        ):
            pc_raster_paths.append(raster_path)
            is_band_pca.append(is_band[i])
            already_read.append(raster_name)
    return (pc_raster_paths, is_band_pca)


def _get_bound(p1: RasterPoint, p2: RasterPoint, is_up_down: bool = False) -> Callable:
    """
    Given two points in plane it computes the straight line through them.
    If we are computing the top or bottom straight lines from our polygon, Y becomes dependent on X and viceversa.

    Straight line through two points equation:
    y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)

    NOTE: (lat, lon) translated to cartesian coordinates can be seen as (y, x)

    """
    if is_up_down:
        return (
            lambda x: (((p2.lat - p1.lat) / (p2.lon - p1.lon)) * (x - p1.lon)) + p1.lat
        )
    else:
        return (
            lambda y: (((p2.lon - p1.lon) / (p2.lat - p1.lat)) * (y - p1.lat)) + p1.lon
        )


def _gps_to_latlon(gps_coords: str) -> RasterPoint:
    """
    Translates GPS coordinates i.e: N22E037 to their corresponding latitude and longitude.
    """
    # TODO: validate `gps_coords` has the appropriate format before parsing it.
    if "W" in gps_coords:
        latitude, longitude = gps_coords.split("W")
        longitude = 0 - float(longitude)
    else:
        latitude, longitude = gps_coords.split("E")
        longitude = float(longitude)

    if "N" in latitude:
        latitude = float(latitude[1:])
    else:
        latitude = 0 - float(latitude[1:])

    return RasterPoint(latitude, longitude)


def _get_corners_raster(
    band_path: Path,
) -> Tuple[RasterPoint, RasterPoint, RasterPoint, RasterPoint]:
    """
    Given a band path, it is opened with rasterio and its bounds are extracted with shapely's transform.

    Returns the raster's corner latitudes and longitudes, along with the band's size.
    """
    final_crs = pyproj.CRS("epsg:4326")

    band = _read_raster(band_path)
    kwargs = _get_kwargs_raster(band_path)
    init_crs = kwargs["crs"]

    project = pyproj.Transformer.from_crs(init_crs, final_crs, always_xy=True).transform

    tl_lon, tl_lat = transform(project, Point(kwargs["transform"] * (0, 0))).bounds[0:2]

    tr_lon, tr_lat = transform(
        project, Point(kwargs["transform"] * (band.shape[2] - 1, 0))
    ).bounds[0:2]

    bl_lon, bl_lat = transform(
        project, Point(kwargs["transform"] * (0, band.shape[1] - 1))
    ).bounds[0:2]

    br_lon, br_lat = transform(
        project, Point(kwargs["transform"] * (band.shape[2] - 1, band.shape[1] - 1))
    ).bounds[0:2]

    tl = RasterPoint(tl_lat, tl_lon)
    tr = RasterPoint(tr_lat, tr_lon)
    bl = RasterPoint(bl_lat, bl_lon)
    br = RasterPoint(br_lat, br_lon)
    return (
        tl,
        tr,
        bl,
        br,
    )


def _sentinel_raster_to_polygon(sentinel_raster_path: str):
    """
    Read a raster and return its bounds as polygon.
    """
    (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ) = _get_corners_raster(sentinel_raster_path)
    sentinel_raster_polygon_json = json.loads(
        f'{{"coordinates": [[[{top_left.lon}, {top_left.lat}], [{top_right.lon}, {top_right.lat}], [{bottom_right.lon}, {bottom_right.lat}], [{bottom_left.lon}, {bottom_left.lat}], [{top_left.lon}, {top_left.lat}]]], "type": "Polygon"}}'
    )
    sentinel_raster_polygon = Polygon.from_bounds(
        top_left.lon, top_left.lat, bottom_right.lon, bottom_right.lat
    )
    return sentinel_raster_polygon, sentinel_raster_polygon_json




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
        label = polygons_in_tile[geometry_id]["label"]

        # Transform point projection to original raster pojection
        project = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), kwargs["crs"], always_xy=True
        ).transform
        tr_point = transform(project, geometry)

        # Get matrix position from the pixel corresponding to a given point coordinates
        row, column = rasterio.transform.rowcol(
            kwargs["transform"], tr_point.x, tr_point.y
        )

        forest_type = polygons_in_tile[geometry_id]["properties"].get("form_arb_d", None)

        label_lon_lat = _label_neighbours(
            kwargs["height"],
            kwargs["width"],
            row,
            column,
            geometry_raw,
            label,
            forest_type,
            label_lon_lat,
        )

    # Get mask from labeled dataset.
    band_mask = label_lon_lat[:, :, 0] == 0

    return band_mask, label_lon_lat


def _check_tiles_not_predicted_in_training(tiles_in_training: List[str], forest_prediction: bool = False):

    minio = MinioConnection()

    if forest_prediction:
        prefix = join(settings.MINIO_DATA_FOLDER_NAME, "forest_classification")
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

    unpredicted_tiles = list(np.setdiff1d(tiles_in_training, predicted_tiles))

    return unpredicted_tiles

def _get_forest_masks(tile: str):
    """
    Get the land cover classification from a certain tile. A mask of pixels that are open forest or closed forest is returned.
    Mask = 1 means closed forest, mask = 2 means open forest
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
        band = band_file.read()

    mask = np.zeros_like(band, dtype=np.uint8)
    mask = np.where(band == 7 , 1, mask) # closed forest, this should be get from .env
    mask = np.where(band == 8 , 2, mask) # open forest, this should be get from .env

    return mask

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
