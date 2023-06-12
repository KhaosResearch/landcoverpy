import json
from itertools import compress
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pyproj
import rasterio
from pymongo.collection import Collection
from rasterio import mask as msk
from rasterio.warp import Resampling, reproject
from shapely.geometry import Point, Polygon
from shapely.ops import transform

from landcoverpy.config import settings
from landcoverpy.exceptions import NoSentinelException
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.minio import MinioConnection
from landcoverpy.rasterpoint import RasterPoint
from landcoverpy.utilities.geometries import (
    _convert_3D_2D,
    _get_corners_geometry,
    _project_shape,
)


def _read_raster(
    band_path: str,
    mask_geometry: dict = None,
    rescale: bool = False,
    path_to_disk: str = None,
    normalize_range: Tuple[float, float] = None,
    to_tif: bool = True,
):
    """
    Reads a raster as a numpy array.
    Parameters:
        band_path (str) : Path of the raster to be read.
        mask_geometry (dict) : If the raster wants to be cropped, a geometry can be provided.
        rescale (bool) : If the raster wans to be rescaled to an spatial resolution of 10m.
        path_to_disk (str) : If the postprocessed (e.g. rescaled, cropped, etc.) raster wants to be saved locally, a path has to be provided
        normalize_range (Tuple[float, float]) : Values mapped to -1 and +1 in normalization. None if the raster doesn't need to be normalized
        to_tif (bool) : If the raster wants to be transformed to a GeoTiff raster (usefull when reading JP2 rasters that can only store natural numbers)

    Returns:
        band (np.ndarray) : The read raster as numpy array

    """
    band_name = _get_raster_name_from_path(str(band_path))
    print(f"Reading raster {band_name}")
    with rasterio.open(band_path) as band_file:
        # Read file
        kwargs = band_file.meta
        band_crs = band_file.crs
        band = band_file.read()

    # Just in case...
    if len(band.shape) == 2:
        band = band.reshape((kwargs["count"], *band.shape))

    # to_float may be better
    if to_tif:
        if kwargs["driver"] == "JP2OpenJPEG":
            band = band.astype(np.float32)
            kwargs["dtype"] = "float32"
            band = np.where(band == 0, np.nan, band)
            kwargs["nodata"] = np.nan
            kwargs["driver"] = "GTiff"

            if path_to_disk is not None:
                path_to_disk = path_to_disk[:-3] + "tif"

    if normalize_range is not None:
        print(f"Normalizing band {band_name}")
        value1, value2 = normalize_range
        band = _normalize(band, value1, value2)

    if rescale:
        band, kwargs = _rescale_band(band, kwargs, 10, band_name)
        

    # Create a temporal memory file to mask the band
    # This is necessary because the band is previously read to scale its resolution
    if mask_geometry:
        print(f"Cropping raster {band_name}")
        projected_geometry = _project_shape(mask_geometry, scs="epsg:4326", dcs=band_crs)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**kwargs) as memfile_band:
                memfile_band.write(band)
                projected_geometry = _convert_3D_2D(projected_geometry)
                masked_band, masked_transform = msk.mask(
                    memfile_band, shapes=[projected_geometry], crop=True, nodata=np.nan
                )
                masked_band = masked_band.astype(np.float32)
                kwargs = memfile_band.meta.copy()
                band = masked_band

        kwargs.update(
            {
                "driver": "GTiff",
                "height": band.shape[1],
                "width": band.shape[2],
                "transform": masked_transform,
            }
        )

    if path_to_disk is not None:
        with rasterio.open(path_to_disk, "w", **kwargs) as dst_file:
            dst_file.write(band)
    return band, kwargs

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
    raise NoSentinelException(f"Either data of product {title} wasn't found in MinIO or 10m bands weren't found for that product.")


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

def _crop_as_sentinel_raster(raster_path: str, sentinel_path: str) -> str:
    """
    Crop a raster merge as a sentinel tile. The resulting image can be smaller than a sentinel tile.

    Since aster products don't exist for areas that don't include any land (tiles with only water),
    the merge of aster products for that area is smaller than the sentinel tile in at least one dimension (missing tile on North and/or  West).
    In the following example the merge product of all the intersecting aster (`+` sign) is smaller in one dimension to the sentinel one (`.` sign):

                                     This 4x4 matrix represents a sentinel tile (center) and the area of the Aster dems needed to cover it.
              |----|                 Legend
              |-..-|                  . = Represent a Sentinel tile
              |+..+|                  + = Merge of several Aster
              |++++|                  - = Missing asters (tile of an area with only of water)

    In the above case, the top left corner of the crop will start on the 3rd row instead of the 2nd, because there is no available aster data to cover it.
    """
    sentinel_kwargs = _get_kwargs_raster(sentinel_path)
    raster_kwargs = _get_kwargs_raster(raster_path)

    # This needs to be corrected on the traslation of the transform matrix
    x_raster, y_raster = raster_kwargs["transform"][2], raster_kwargs["transform"][5]
    x_sentinel, y_sentinel = (
        sentinel_kwargs["transform"][2],
        sentinel_kwargs["transform"][5],
    )
    # Use the smaller value (the one to the bottom in the used CRS) for the transform, to reproject to the intersection
    y_transform_position = (
        raster_kwargs["transform"][5]
        if y_raster < y_sentinel
        else sentinel_kwargs["transform"][5]
    )
    # Use the bigger value (the one to the right in the used CRS) for the transform, to reproject to the intersection
    x_transform_position = (
        raster_kwargs["transform"][2]
        if x_raster > x_sentinel
        else sentinel_kwargs["transform"][2]
    )

    _, sentinel_polygon = _sentinel_raster_to_polygon(sentinel_path)
    cropped_raster, _ = _read_raster(
        raster_path, mask_geometry=sentinel_polygon, rescale=False
    )
    cropped_raster_kwargs = raster_kwargs.copy()
    cropped_raster_kwargs["transform"] = rasterio.Affine(
        raster_kwargs["transform"][0],
        0.0,
        x_transform_position,
        0.0,
        raster_kwargs["transform"][4],
        y_transform_position,
    )
    cropped_raster_kwargs.update(
        {
            "width": cropped_raster.shape[2],
            "height": cropped_raster.shape[1],
        }
    )

    dst_kwargs = sentinel_kwargs.copy()
    dst_kwargs["dtype"] = cropped_raster_kwargs["dtype"]
    dst_kwargs["nodata"] = cropped_raster_kwargs["nodata"]
    dst_kwargs["driver"] = cropped_raster_kwargs["driver"]
    dst_kwargs["transform"] = rasterio.Affine(
        sentinel_kwargs["transform"][0],
        0.0,
        x_transform_position,
        0.0,
        sentinel_kwargs["transform"][4],
        y_transform_position,
    )

    with rasterio.open(raster_path, "w", **dst_kwargs) as dst:
        reproject(
            source=cropped_raster,
            destination=rasterio.band(dst, 1),
            src_transform=cropped_raster_kwargs["transform"],
            src_crs=cropped_raster_kwargs["crs"],
            dst_transform=dst_kwargs["transform"],
            dst_crs=sentinel_kwargs["crs"],
            resampling=Resampling.nearest,
        )

    # For prediction, raster is filled with 0 to have equal dimensions to the Sentinel product (aster products in water are always 0).
    # This is made only for prediction because in training pixels are obtained using latlong, it will be a waste of time.
    # In prediction, the same dimensions are needed because the whole product is converted to a flattered array, then concatenated to a big dataframe.

    if (y_transform_position < y_sentinel) or (x_transform_position > x_sentinel):

        spatial_resolution = sentinel_kwargs["transform"][0]
        
        with rasterio.open(raster_path) as raster_file:
            cropped_raster_kwargs = raster_file.meta
            cropped_raster = raster_file.read(1) 

        row_difference = int((y_sentinel - y_transform_position)/spatial_resolution)
        column_difference = int((x_sentinel - x_transform_position)/spatial_resolution)
        cropped_raster = np.roll(cropped_raster, (row_difference,-column_difference), axis=(0,1))
        cropped_raster[:row_difference,:] = 0
        cropped_raster[:,:column_difference] = 0

        cropped_raster_kwargs["transform"] = rasterio.Affine(
            sentinel_kwargs["transform"][0],
            0.0,
            sentinel_kwargs["transform"][2],
            0.0,
            sentinel_kwargs["transform"][4],
            sentinel_kwargs["transform"][5],
        )

        with rasterio.open(raster_path, "w", **cropped_raster_kwargs) as dst:
            dst.write(cropped_raster.reshape(1,cropped_raster.shape[0],-1))

    return raster_path


def _rescale_band(
    band: np.ndarray,
    kwargs: dict,
    spatial_resol: int,
    band_name: str
):
    img_resolution = kwargs["transform"][0]

    if img_resolution != spatial_resol:
        scale_factor = img_resolution / spatial_resol

        new_kwargs = kwargs.copy()
        new_kwargs["height"] = int(kwargs["height"] * scale_factor)
        new_kwargs["width"] = int(kwargs["width"] * scale_factor)
        new_kwargs["transform"] = rasterio.Affine(
        spatial_resol, kwargs["transform"][1], kwargs["transform"][2], kwargs["transform"][3], -spatial_resol, kwargs["transform"][5])

        rescaled_raster = np.ndarray(
            shape=(kwargs["count"], new_kwargs["height"], new_kwargs["width"]), dtype=np.float32)

        print(f"Rescaling raster {band_name}, from: {img_resolution}m to {str(spatial_resol)}.0m")
        reproject(
            source=band,
            destination=rescaled_raster,
            src_transform=kwargs["transform"],
            src_crs=kwargs["crs"],
            dst_resolution=(new_kwargs["width"], new_kwargs["height"]),
            dst_transform=new_kwargs["transform"],
            dst_crs=new_kwargs["crs"],
            resampling=Resampling.nearest,
        )
        band = rescaled_raster
        kwargs = new_kwargs

    return band, kwargs

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

def _get_corners_raster(
    band_path: Path,
) -> Tuple[RasterPoint, RasterPoint, RasterPoint, RasterPoint]:
    """
    Given a band path, it is opened with rasterio and its bounds are extracted with shapely's transform.

    Returns the raster's corner latitudes and longitudes, along with the band's size.
    """
    final_crs = pyproj.CRS("epsg:4326")

    band, kwargs = _read_raster(band_path)
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
    except Exception:
        normalized_matrix = matrix
    return normalized_matrix
