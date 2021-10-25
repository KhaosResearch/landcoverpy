from raster_point import RasterPoint

import glob
import os
import xml.dom.minidom as minidom
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pyproj
import rasterio
from rasterio import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point
from shapely.ops import transform


def get_crs(xml_path: Path) -> str:
    """
    Given a product's metadata XML path, returns its coordinate system.
    """
    xml = minidom.parse(xml_path)
    return xml.getElementsByTagName("HORIZONTAL_CS_CODE")[0].lastChild.nodeValue


def get_sun(xml_path: Path) -> Tuple[float, float]:
    """
    Given a product's metadata XML path, returns its azimuth and height.
    """
    xml = minidom.parse(xml_path)
    azimuth = float(xml.getElementsByTagName("AZIMUTH_ANGLE")[0].lastChild.nodeValue)
    zenith = float(xml.getElementsByTagName("ZENITH_ANGLE")[0].lastChild.nodeValue)
    height = 90 - zenith

    return azimuth, height


def get_corners(
    band_path: Path, scs: str = "epsg:32629", dcs: str = "epsg:4326"
) -> Tuple[RasterPoint, RasterPoint, RasterPoint, RasterPoint]:
    """
    Given a band path, it is opened with rasterio and its bounds are extracted with shapely's transform.

    Returns the raster's corner latitudes and longitudes, along with the band's size.
    """
    init_crs = pyproj.CRS(scs)
    final_crs = pyproj.CRS(dcs)

    project = pyproj.Transformer.from_crs(init_crs, final_crs, always_xy=True).transform

    with rasterio.open(band_path) as bnd:
        band = bnd.read(1).astype(np.float32)
        kwargs = bnd.meta

    tl_lon, tl_lat = transform(project, Point(kwargs["transform"] * (0, 0))).bounds[0:2]

    tr_lon, tr_lat = transform(
        project, Point(kwargs["transform"] * (band.shape[1] - 1, 0))
    ).bounds[0:2]

    bl_lon, bl_lat = transform(
        project, Point(kwargs["transform"] * (0, band.shape[0] - 1))
    ).bounds[0:2]

    br_lon, br_lat = transform(
        project, Point(kwargs["transform"] * (band.shape[1] - 1, band.shape[0] - 1))
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


def get_bound(p1: RasterPoint, p2: RasterPoint, is_up_down: bool = False) -> Callable:
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


def gps_to_latlon(gps_coords: str) -> RasterPoint:
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


def gather_aster(
    dem_path: Path,
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

    for dem in os.listdir(dem_path):
        bl_dem = gps_to_latlon(dem)

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
                    overlapping_dem.append(dem_path / dem)

    return overlapping_dem


def merge_dem(dem_paths: List[Path], outfile: Path) -> Path:
    """
    Given a list of DEM paths and the corresponding product's geometry, merges them into a single raster.

    Returns the path to the merged DEM.
    """

    dems = [
        glob.glob(str(dm) + f"/*_{str(dm).split('/')[-1]}.sdat")[0] for dm in dem_paths
    ]
    dem, dem_transform = merge.merge(dems)
    out_meta = {
        "driver": "SAGA",
        "dtype": "float32",
        "count": 1,
        "height": dem.shape[1],
        "width": dem.shape[2],
        "crs": pyproj.CRS.from_epsg(4326),
        "transform": dem_transform,
    }

    if not os.path.exists(os.path.dirname(outfile)):
        try:
            os.makedirs(os.path.dirname(outfile))
        except OSError as err:  # Guard against race condition
            print("Could not create merged DEM file path: ", err)

    with rasterio.open(outfile, "w", **out_meta) as dm:
        dm.write(dem)

    return outfile


def reproject_dem(dem_path: Path, dst_crs: str) -> Path:
    """
    Given a DEM model and a destination CRS, performs a reprojection and resampling of the original CRS.

    Returns the reprojected DEM path.
    """
    reprojected_path = Path(str(dem_path).split(".sdat")[0] + "_" + dst_crs + ".sdat")

    with rasterio.open(dem_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
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
