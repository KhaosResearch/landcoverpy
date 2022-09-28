import numpy as np
from rasterio import mask as msk
from rasterio.warp import Resampling, reproject
from shapely.geometry import Point
from shapely.ops import transform

from etc_workflow.execution_mode import ExecutionMode

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
        destination_crs = band_file.crs
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
        projected_geometry = _project_shape(mask_geometry, dcs=destination_crs)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**kwargs) as memfile_band:
                memfile_band.write(band)
                projected_geometry = _convert_3D_2D(projected_geometry)
                masked_band, _ = msk.mask(
                    memfile_band, shapes=[projected_geometry], crop=True, nodata=np.nan
                )
                masked_band = masked_band.astype(np.float32)
                band = masked_band

        new_kwargs = kwargs.copy()
        corners = _get_corners_geometry(mask_geometry)
        top_left_corner = corners["top_left"]
        top_left_corner = (top_left_corner[1], top_left_corner[0])
        project = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), new_kwargs["crs"], always_xy=True
        ).transform
        top_left_corner = transform(project, Point(top_left_corner))
        new_kwargs["transform"] = rasterio.Affine(
            new_kwargs["transform"][0],
            0.0,
            top_left_corner.x,
            0.0,
            new_kwargs["transform"][4],
            top_left_corner.y,
        )
        new_kwargs["width"] = band.shape[2]
        new_kwargs["height"] = band.shape[1]
        kwargs = new_kwargs

    if path_to_disk is not None:
        with rasterio.open(path_to_disk, "w", **kwargs) as dst_file:
            dst_file.write(band)
    return band

def _crop_as_sentinel_raster(execution_mode: ExecutionMode, raster_path: str, sentinel_path: str) -> str:
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
    cropped_raster = _read_raster(
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

    if execution_mode != ExecutionMode.TRAINING:

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
        spatial_resol, 0.0, kwargs["transform"][2], 0.0, -spatial_resol, kwargs["transform"][5])

        rescaled_raster = np.ndarray(
            shape=(new_kwargs["height"], new_kwargs["width"]), dtype=np.float32)

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
        band = rescaled_raster.reshape((new_kwargs["count"], *rescaled_raster.shape))
        kwargs = new_kwargs

    return band, kwargs