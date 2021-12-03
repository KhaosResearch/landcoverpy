import glob
import os
from pathlib import Path
import re
from typing import Callable, List, Tuple

import shutil
import numpy as np
import pyproj
import rasterio
from rasterio import merge
from shapely.geometry import Point
from shapely.ops import transform
from pymongo.collection import Collection
from minio import Minio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from utils import get_product_rasters_paths, get_minio, read_raster, _get_kwargs_raster, get_raster_filename_from_path, connect_mongo_products_collection,connect_mongo_composites_collection
from itertools import compress
from config import settings
import json

class RasterPoint:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return f"{self.lat}, {self.lon}"


def _get_corners(
    band_path: Path,  dcs: str = "epsg:4326"
) -> Tuple[RasterPoint, RasterPoint, RasterPoint, RasterPoint]:
    """
    Given a band path, it is opened with rasterio and its bounds are extracted with shapely's transform.

    Returns the raster's corner latitudes and longitudes, along with the band's size.
    """
    final_crs = pyproj.CRS(dcs)


    band = read_raster(band_path,no_data_value=-1)
    kwargs = _get_kwargs_raster(band_path)
    init_crs = kwargs['crs']

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
                if "slope.sdat" == file_object.split('/')[-1]:
                    slope.append(str(Path(outpath, file_object)))
                elif "aspect.sdat" == file_object.split('/')[-1]:
                    aspect.append(str(Path(outpath, file_object)))

        slope, slope_transform = merge.merge(slope)
        aspect, aspect_transform = merge.merge(aspect)

        out_slope_meta = {
            "driver": "SAGA",
            "dtype": "float32",
            "count": 1,
            "height": slope.shape[1],
            "width": slope.shape[2],
            "crs": pyproj.CRS.from_epsg(4326),
            "transform": slope_transform,
        }

        out_aspect_meta = {
            "driver": "SAGA",
            "dtype": "float32",
            "count": 1,
            "height": aspect.shape[1],
            "width": aspect.shape[2],
            "crs": pyproj.CRS.from_epsg(4326),
            "transform": aspect_transform,
        }
        outpath_slope = str(Path(outpath, "slope.sdat"))
        outpath_aspect = str(Path(outpath, "aspect.sdat"))

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
    reprojected_path = Path(str(dem_path).split(".sdat")[0] + "_r.sdat")

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

def dem_to_sentinel_raster(dem_path: str, sentinel_raster_path: str) -> str:
    
    sentinel_raster_kwargs = _get_kwargs_raster(sentinel_raster_path)
    dem_kwargs = _get_kwargs_raster(dem_path)

    spatial_resolution = sentinel_raster_kwargs['transform'][0]
    top_left_corner = (sentinel_raster_kwargs['transform'][2],sentinel_raster_kwargs['transform'][5])
    top_right_corner = (top_left_corner[0] + spatial_resolution * sentinel_raster_kwargs['width'],top_left_corner[1])
    bottom_left_corner = (top_left_corner[0], top_left_corner[1] - spatial_resolution * sentinel_raster_kwargs['height'])
    bottom_right_corner = (top_right_corner[0], bottom_left_corner[1])

    project = pyproj.Transformer.from_crs(sentinel_raster_kwargs['crs'], pyproj.CRS.from_epsg(4326), always_xy=True).transform

    top_left_corner = transform(project, Point(top_left_corner))
    top_right_corner = transform(project, Point(top_right_corner))
    bottom_left_corner = transform(project, Point(bottom_left_corner))
    bottom_right_corner = transform(project, Point(bottom_right_corner))

    sentinel_raster_polygon = json.loads(f'{{"coordinates": [[[{top_left_corner.x}, {top_left_corner.y}], [{top_right_corner.x}, {top_right_corner.y}], [{bottom_right_corner.x}, {bottom_right_corner.y}], [{bottom_left_corner.x}, {bottom_left_corner.y}], [{top_left_corner.x}, {top_left_corner.y}]]], "type": "Polygon"}}')
    cropped_dem = read_raster(dem_path, mask_geometry=sentinel_raster_polygon, rescale=False, no_data_value=-1)
    cropped_dem_kwargs = dem_kwargs.copy()
    cropped_dem_kwargs['transform'] = dem_kwargs['transform']
    cropped_dem_kwargs['transform'] = rasterio.Affine(dem_kwargs['transform'][0], 0.0, sentinel_raster_kwargs["transform"][2], 0.0,dem_kwargs['transform'][4] , sentinel_raster_kwargs["transform"][5])
    cropped_dem_kwargs.update({'width': cropped_dem.shape[1], 'height': cropped_dem.shape[1], })
    
    dst_kwargs = sentinel_raster_kwargs.copy()
    dst_kwargs['dtype'] = cropped_dem_kwargs['dtype']
    dst_kwargs['nodata'] = cropped_dem_kwargs['nodata']
    dst_kwargs['driver'] = cropped_dem_kwargs['driver']

    with rasterio.open(dem_path, "w", **dst_kwargs) as dst:
        reproject(
            source=cropped_dem,
            destination=rasterio.band(dst, 1),
            src_transform=cropped_dem_kwargs['transform'],
            src_crs=cropped_dem_kwargs['crs'],
            dst_resolution=(sentinel_raster_kwargs['width'], sentinel_raster_kwargs['height']),
            dst_transform=sentinel_raster_kwargs['transform'],
            dst_crs=sentinel_raster_kwargs['crs'],
            resampling=Resampling.nearest,
        )
    
    return dem_path

def get_slope_aspect_from_tile(tile: str, mongo_collection: Collection,minio_client: Minio, minio_bucket_product: str, minio_bucket_aster: str):

    product_metadata = mongo_collection.find_one({
                "title": {
                    "$regex": f"_T{tile}_"
                }
    })
    product_title = product_metadata['title']
    product_path = str(Path(settings.TMP_DIR,product_title))
    rasters_paths, is_band = get_product_rasters_paths(product_metadata, minio_client, minio_bucket_product)
    sample_band_path_minio = list(compress(rasters_paths, is_band))[0]
    sample_band_path = str(Path(product_path, get_raster_filename_from_path(sample_band_path_minio)))
    minio_client.fget_object(
        bucket_name=minio_bucket_product,
        object_name=sample_band_path_minio,
        file_path=str(sample_band_path),
    )

    (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ) = _get_corners(sample_band_path)
    
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
        outpath= product_path,
        minio_client=minio_client
    )

    kwargs = _get_kwargs_raster(sample_band_path)
    r_slope = _reproject_dem(slope_path, str(kwargs['crs']))
    r_aspect = _reproject_dem(aspect_path, str(kwargs['crs']))

    r_slope = dem_to_sentinel_raster(r_slope, sample_band_path)
    r_aspect = dem_to_sentinel_raster(r_aspect, sample_band_path)


    return r_slope , r_aspect
