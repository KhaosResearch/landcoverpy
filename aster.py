import os
from pathlib import Path
from typing import Callable, List
import rasterio
from rasterio import merge
from pymongo.collection import Collection
from minio import Minio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from config import settings
from rasterpoint import RasterPoint
from utils import (
    _get_kwargs_raster, 
    _get_corners_raster,
    _gps_to_latlon,
    _get_bound,
    crop_as_sentinel_raster,
    download_sample_band,
)


def _gather_aster(
    minio_client: Minio, 
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
        dem_object = dem.object_name.split('/')[0]
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


def _merge_dem(dem_paths: List[Path], outpath: str, minio_client: Minio) -> Path:
    """
    Given a list of DEM paths and the corresponding product's geometry, merges them into a single raster.

    Returns the path to the merged DEM.
    """
    slope = []
    aspect = []
    if not len(dem_paths) == 0:
        for dem_path in dem_paths:
            for file in minio_client.list_objects(settings.MINIO_BUCKET_NAME_ASTER, prefix=dem_path, recursive=True):
                file_object = file.object_name
                minio_client.fget_object(
                    bucket_name=settings.MINIO_BUCKET_NAME_ASTER,
                    object_name=file_object,
                    file_path=str(Path(outpath, file_object))
                )
                if "slope.tif" == file_object.split('/')[-1]:
                    slope.append(str(Path(outpath, file_object)))
                elif "aspect.tif" == file_object.split('/')[-1]:
                    aspect.append(str(Path(outpath, file_object)))

        out_slope_meta = _get_kwargs_raster(slope[0])
        out_aspect_meta = _get_kwargs_raster(aspect[0])

        slope, slope_transform = merge.merge(slope)
        aspect, aspect_transform = merge.merge(aspect)

        out_slope_meta["height"] = slope.shape[1]
        out_slope_meta["width"] = slope.shape[2]
        out_slope_meta["transform"] = slope_transform

        out_aspect_meta["height"] = aspect.shape[1]
        out_aspect_meta["width"] = aspect.shape[2]
        out_aspect_meta["transform"] = aspect_transform

        outpath_slope = str(Path(outpath, "slope.tif"))
        outpath_aspect = str(Path(outpath, "aspect.tif"))

        if not os.path.exists(os.path.dirname(outpath_slope)):
            try:
                os.makedirs(os.path.dirname(outpath_slope))
            except OSError as err:  # Guard against race condition
                print("Could not create merged DEM file path: ", err)

        if not os.path.exists(os.path.dirname(outpath_aspect)):
            try:
                os.makedirs(os.path.dirname(outpath_aspect))
            except OSError as err:  # Guard against race condition
                print("Could not create merged DEM file path: ", err)

        with rasterio.open(outpath_slope, "w", **out_slope_meta) as dm:
            dm.write(slope)
        with rasterio.open(outpath_aspect, "w", **out_aspect_meta) as dm:
            dm.write(aspect)

        return outpath_slope, outpath_aspect
    else:
        print("Any dem found for this product")

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
            {"driver": "GTiff" ,"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(reprojected_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
    return reprojected_path

def get_slope_aspect_from_tile(tile: str, mongo_collection: Collection,minio_client: Minio, minio_bucket_aster: str):
    '''
    Create both aspect and slope rasters merging aster products and proyecting them to sentinel rasters.
    '''

    sample_band_path = download_sample_band(tile, minio_client, mongo_collection)

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
        minio_client, minio_bucket_aster, left_bound, right_bound, upper_bound, lower_bound
    )
    
    print(f"Obtaining slope and aspect data of tile {tile} using {overlapping_dem} aster products")
    slope_path, aspect_path = _merge_dem(
        dem_paths=overlapping_dem,
        outpath= str(Path(settings.TMP_DIR)),
        minio_client=minio_client
    )

    kwargs = _get_kwargs_raster(sample_band_path)
    r_slope_path = _reproject_dem(slope_path, str(kwargs['crs']))
    r_aspect_path = _reproject_dem(aspect_path, str(kwargs['crs']))

    r_slope_path = crop_as_sentinel_raster(r_slope_path, sample_band_path)
    r_aspect_path = crop_as_sentinel_raster(r_aspect_path, sample_band_path)


    return r_slope_path , r_aspect_path
