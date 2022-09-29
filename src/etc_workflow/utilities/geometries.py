import json
import warnings
from typing import Callable

import geopandas as gpd
import numpy as np
import pyproj
from bs4 import BeautifulSoup
from mgrs import MGRS
from functools import partial
from sentinelsat.sentinel import  read_geojson
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform
from zipfile import ZipFile

from etc_workflow.rasterpoint import RasterPoint

def _kmz_to_geojson(kmz_file: str) -> str:
    """
    Transform a kmz file to a geojson file
    """
    import fiona
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    geojson_file = kmz_file[:-4] + ".geojson"
    with ZipFile(kmz_file, "r") as zip_in:
        zip_in.extractall("./databases/")
    df = gpd.read_file(filename="./databases/doc.kml", driver="KML")
    df.to_file(geojson_file, driver="GeoJSON")

    _postprocess_geojson_file(geojson_file)

    return geojson_file

def _postprocess_geojson_file(geojson_file: str):
    """
    Postprocess a geojson that comes from a kmz, transform its html table to a dictionary
    """
    geojson = read_geojson(geojson_file)

    for feature in geojson['features']:

        propery = feature["properties"]

        html_table = propery["Description"]

        if 'html' in html_table:
            html_table_splitted = html_table.split('</td> </tr> <tr> <td> ')
            html_table_without_header = html_table_splitted[1]
            html_table_without_header_splitted = html_table_without_header.split('</td> </tr> </table> </body>')
            content = html_table_without_header_splitted[0]
            
            parser_object = BeautifulSoup(content, 'lxml') 
            key_value_list = parser_object.find_all('td') 
            key_value_list_text=[element.get_text() for element in key_value_list]

            key_value_list_text = np.array(key_value_list_text)
            pairs= list(range(0,len(key_value_list_text),2))
            evens= list(range(1,len(key_value_list_text),2))
            keys=key_value_list_text[pairs]
            values=key_value_list_text[evens]

        else:
            keys=""
            values=""

        parsed_description={}
        for c in range(len(values)):
            parsed_description[keys[c]]= values[c]
            

        del propery["Description"]
        propery.update(parsed_description)

    with open(geojson_file, "w", encoding='utf8') as f:
        json.dump(geojson, f, ensure_ascii=False)

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

