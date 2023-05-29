from os.path import join

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from landcoverpy.config import settings
from landcoverpy.minio import MinioConnection


import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def transform_to_latlon(input_file, output_file):
    with rasterio.open(input_file) as src:
        dst_crs = 'EPSG:4326' 

        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

def rescale_predictions(res: int):
    """
    Rescale all predicted tiles to another spatial resolution.

    Parameters:
        res (int): Out spatial resolution
    """

    minio_client = MinioConnection()

    # List all classifications in minio
    classified_tiles_cursor = minio_client.list_objects(
        bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
        prefix=join(settings.MINIO_DATA_FOLDER_NAME, "forest-0-6", "forest_classification")
    )

    # For each image
    for classified_tile_object in classified_tiles_cursor:
        classification_path = classified_tile_object.object_name
        
        classification_filename = classification_path.split("/")[-1]
        local_classification_path = join(
            settings.TMP_DIR, "classifications_10m", classification_filename
        )

        print(classification_filename)

        minio_client.fget_object(
            bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
            object_name=classification_path,
            file_path=local_classification_path,
        )

        with rasterio.open(local_classification_path) as raster_file:
            in_kwargs = raster_file.meta
            raster = raster_file.read()
            out_kwargs = in_kwargs.copy()
            out_kwargs["transform"] = rasterio.Affine(
                res,
                0.0,
                in_kwargs["transform"][2],
                0.0,
                -res,
                in_kwargs["transform"][5],
            )

            out_kwargs["height"] = int(in_kwargs["height"] * (10 / res))
            out_kwargs["width"] = int(in_kwargs["width"] * (10 / res))
            out_kwargs["dtype"] = np.uint8

            rescaled_raster = np.ndarray(
                shape=(out_kwargs["height"], out_kwargs["width"]), dtype=np.uint8
            )
            reproject(
                source=raster,
                destination=rescaled_raster,
                src_transform=in_kwargs["transform"],
                src_crs=in_kwargs["crs"],
                dst_resolution=(out_kwargs["width"], out_kwargs["height"]),
                dst_transform=out_kwargs["transform"],
                dst_crs=out_kwargs["crs"],
                resampling=Resampling.nearest,
            )

            raster = rescaled_raster.reshape(
                (out_kwargs["count"], *rescaled_raster.shape)
            )

        local_rescaled_path = join(
            settings.TMP_DIR,
            classification_path.split("/")[-1],
        )

        with rasterio.open(local_rescaled_path, "w", **out_kwargs) as dst_file:
            dst_file.write(raster)

        transform_to_latlon(local_rescaled_path, local_rescaled_path)

        minio_client.fput_object(
            bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
            object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/forest-0-6/{res}m/{classification_filename}",
            file_path=local_rescaled_path,
            content_type="image/tif",
        )