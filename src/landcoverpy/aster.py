import os
from pathlib import Path
from typing import Callable, List

import numpy as np
import rasterio
from pymongo.collection import Collection
from rasterio import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from landcoverpy.config import settings
from landcoverpy.exceptions import NoAsterException
from landcoverpy.minio import MinioConnection
from landcoverpy.rasterpoint import RasterPoint
from landcoverpy.utilities.geometries import _get_bound, _gps_to_latlon
from landcoverpy.utilities.raster import (
    _crop_as_sentinel_raster,
    _download_sample_band_by_tile,
    _get_corners_raster,
    _get_kwargs_raster,
)


def _get_bucket_by_name(dem_name: str) -> str:
    """
    Deppending of the raster read, a different minio bucket will be returned.
    This is needed because DEM rasters are stored in a different minio bucket than aspect and slope rasters.
    """
    if dem_name == "dem":
        bucket = settings.MINIO_BUCKET_NAME_DEM
    else:
        bucket = settings.MINIO_BUCKET_NAME_ASTER
    return bucket


def _gather_aster(
    minio_client: MinioConnection,
    minio_bucket: str,
    left_bound: Callable,
    right_bound: Callable,
    upper_bound: Callable,
    lower_bound: Callable,
) -> List[Path]:
    """
    Given the four straight lines bounds from a product and the asters' path, retrieves those aster DEMs which have pixels inside the product.

    Each aster's DEM is named after its location in the grid (N/W), the name represents the bottom left corner of the cell.

    Returns a list of aster DEMs strings.
    """
    overlapping_dem = []

    for dem in minio_client.list_objects(minio_bucket):
        dem_object = dem.object_name.split("/")[0]
        bl_dem = _gps_to_latlon(dem_object)
        # TODO: not neccessary for our case study, however latitude and longitude maximum and minimum values might cause trouble
        # - Latitude: [-90, 90]
        # - Longitude: [-180, 180]
        # DEM in those borders need their coordinates to be corrected, as a longitude of -181 would translate to 180.

        tl_dem = RasterPoint(bl_dem.lat + 1, bl_dem.lon)
        br_dem = RasterPoint(bl_dem.lat, bl_dem.lon + 1)
        tr_dem = RasterPoint(bl_dem.lat + 1, bl_dem.lon + 1)

        dem_corners = [bl_dem, tl_dem, br_dem, tr_dem]

        for dc in dem_corners:
            if left_bound(dc.lat) <= dc.lon <= right_bound(dc.lat):
                if lower_bound(dc.lon) <= dc.lat <= upper_bound(dc.lon):
                    overlapping_dem.append(dem.object_name)

    return overlapping_dem


def _merge_dem(
    dem_paths: List[Path], outpath: str, minio_client: MinioConnection, dem_name: str
) -> Path:
    """
    Given a list of DEM paths and the corresponding product's geometry, merges them into a single raster.

    Returns the path to the merged DEM.
    """
    dem = []
    bucket = _get_bucket_by_name(dem_name)
    for dem_path in dem_paths:
        for file in minio_client.list_objects(
            bucket, prefix=dem_path, recursive=True
        ):
            file_object = file.object_name
            minio_client.fget_object(
                bucket_name=bucket,
                object_name=file_object,
                file_path=str(Path(outpath, file_object)),
            )
            if file_object.endswith(f"{dem_name}.tif"):
                dem.append(str(Path(outpath, file_object)))

    out_dem_meta = _get_kwargs_raster(dem[0])
    dem, dem_transform = merge.merge(dem)
    out_dem_meta["driver"] = "GTiff"
    out_dem_meta["dtype"] = "float32"
    out_dem_meta["height"] = dem.shape[1]
    out_dem_meta["width"] = dem.shape[2]
    out_dem_meta["transform"] = dem_transform
    out_dem_meta["nodata"] = np.nan

    outpath_dem = str(Path(outpath, f"{dem_name}.tif"))

    if not os.path.exists(os.path.dirname(outpath_dem)):
        os.makedirs(os.path.dirname(outpath_dem))

    with rasterio.open(outpath_dem, "w", **out_dem_meta) as dm:
        dm.write(dem)

    return outpath_dem


def _reproject_dem(dem_path: Path, dst_crs: str) -> Path:
    """
    Given a DEM model and a destination CRS, performs a reprojection and resampling of the original CRS.

    Returns the reprojected DEM path.
    """
    reprojected_path = Path(str(dem_path).split(".tif")[0] + "_r.tif")

    with rasterio.open(dem_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "driver": "GTiff",
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": np.nan,
            }
        )

        with rasterio.open(reprojected_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
    return reprojected_path


def get_dem_from_tile(
    tile: str, mongo_collection: Collection, minio_client: MinioConnection, dem_name: str
):
    """
    Create both aspect and slope rasters merging aster products and proyecting them to sentinel rasters.
    """
    bucket = _get_bucket_by_name(dem_name)

    sample_band_path = _download_sample_band_by_tile(tile, minio_client, mongo_collection)

    (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ) = _get_corners_raster(sample_band_path)

    left_bound = _get_bound(bottom_left, top_left)
    right_bound = _get_bound(bottom_right, top_right)
    upper_bound = _get_bound(top_left, top_right, is_up_down=True)
    lower_bound = _get_bound(bottom_left, bottom_right, is_up_down=True)

    overlapping_dem = _gather_aster(
        minio_client, bucket, left_bound, right_bound, upper_bound, lower_bound
    )
    if overlapping_dem == []:
        raise NoAsterException(f"There is no Aster products related to tile {tile}, probably a full-water tile.")

    print(
        f"Obtaining {dem_name} data of tile {tile} using {overlapping_dem} aster products"
    )
    dem_path = _merge_dem(
        dem_paths=overlapping_dem,
        outpath=str(Path(settings.TMP_DIR)),
        minio_client=minio_client,
        dem_name=dem_name,
    )

    kwargs = _get_kwargs_raster(sample_band_path)
    r_dem_path = _reproject_dem(dem_path, str(kwargs["crs"]))

    r_dem_path = _crop_as_sentinel_raster(r_dem_path, sample_band_path)

    return r_dem_path
