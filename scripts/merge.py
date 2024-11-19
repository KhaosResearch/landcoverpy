import os
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from rasterio.mask import mask

from landcoverpy.minio import MinioConnection

from dotenv import load_dotenv

load_dotenv()

def reproject_raster_to_4326(local_path: str) -> tuple[np.ndarray, dict]:
    """
    Reprojects a single-band raster file to EPSG:4326 if necessary and saves it back to the same path.

    This function checks the CRS (Coordinate Reference System) of the given raster file and, if it is not already
    in EPSG:4326, reprojects it to this CRS. The reprojected raster is saved to the same file path.

    Parameters:
    local_path : str
        The path to the local single-band raster file that will be reprojected.

    Returns:
    tuple
        A tuple containing:
        - reprojected_band (numpy.ndarray): The reprojected raster band as a NumPy array.
        - reprojected_meta (dict): Metadata dictionary for the reprojected raster.
    
    Raises:
    ValueError
        If the input raster file has more than one band.
    """
    with rasterio.open(local_path) as src:
        if src.count != 1:
            raise ValueError("The raster file must be single-band.")

        if src.crs.to_epsg() == 4326:
            band = src.read(1)
            return band, src.meta

        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
        )

        reprojected_meta = src.meta.copy()
        reprojected_meta.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        reprojected_band = np.empty((height, width), dtype=src.meta['dtype'])

        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected_band,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.nearest
        )

        with rasterio.open(local_path, 'w', **reprojected_meta) as dst:
            dst.write(reprojected_band, 1)

        return reprojected_band, reprojected_meta

def merge_rasters(local_paths: list[Path]) -> Path:
    """
    Merges a list of raster files, ensuring that they are reprojected to EPSG:4326, and saves the result to a new file.

    This function reprojects each raster file to EPSG:4326, merges them into a single raster, and saves the merged
    raster to a new file in the directory specified by the "TMP_DIR" environment variable.

    Parameters:
    local_paths : list of Path
        A list of Path objects pointing to the local raster files to be merged. All rasters must be single-band.

    Returns:
    Path
        The path to the merged raster file.

    Raises:
    ValueError
        If any input raster file is not single-band.
    FileNotFoundError
        If any of the input file paths do not exist.
    """
    for local_path in local_paths:
        if not local_path.exists():
            raise FileNotFoundError(f"The file {local_path} does not exist.")
        if not local_path.is_file():
            raise ValueError(f"The path {local_path} is not a valid file.")

    tmp_dir = os.environ.get("TMP_DIR")
    if not tmp_dir:
        raise EnvironmentError("The 'TMP_DIR' environment variable is not set.")
    output_path = Path(tmp_dir, f"merged_{local_paths[0].name}")

    for local_path in local_paths:
        reproject_raster_to_4326(local_path)

    src_files_to_merge = [rasterio.open(path) for path in local_paths]
    out_image, out_transform = merge(src_files_to_merge)

    out_meta = src_files_to_merge[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "crs": src_files_to_merge[0].crs
    })

    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(out_image)

    for src in src_files_to_merge:
        src.close()
    for local_path in local_paths:
        local_path.unlink()

    return output_path

def reproject_and_crop_raster(source_path, mask_path, output_cropped_path):
    """
    Reprojects and crops a raster based on the bounds and CRS of a reference raster.
    
    Parameters:
    source_path (str or Path): Path to the raster to be reprojected and cropped.
    mask_path (str or Path): Path to the reference raster used for cropping.
    output_cropped_path (str or Path): Path where the cropped output raster will be saved.
    """
    
    # Open the reference raster to obtain its bounds and CRS
    with rasterio.open(mask_path) as ref_raster:
        ref_bounds = ref_raster.bounds
        ref_crs = ref_raster.crs
    
    # Reproject the source raster to match the CRS of the mask
    with rasterio.open(source_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, ref_crs, src.width, src.height, *src.bounds
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "crs": ref_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Temporary file for the reprojected raster
        temp_reprojected_path = Path(output_cropped_path).with_name("temp_reprojected.tif")

        # Save the reprojected raster
        with rasterio.open(temp_reprojected_path, "w", **out_meta) as dest:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dest, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )

    # Convert the BoundingBox of mask.jp2 to a Shapely box and then to GeoJSON format
    geom = [box(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top).__geo_interface__]

    # Crop the reprojected raster using the bounds of mask.jp2
    with rasterio.open(temp_reprojected_path) as reprojected:
        out_image, out_transform = mask(reprojected, geom, crop=True)
        
        # Update metadata to match the cropped raster
        out_meta = reprojected.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save the cropped raster
        with rasterio.open(output_cropped_path, "w", **out_meta) as dest:
            dest.write(out_image)

def main(interpolating: bool = False, crop: bool = False):
    """
    Reads all chunks of raster files from the MinIO bucket, merges them, and saves the result back to the bucket.
    Reads from the "rasters/{feature}/" directory and writes to "rasters/merged/feature.tif".
    """

    minio_client = MinioConnection()
    
    objects = minio_client.list_objects("pnoa-lidar", prefix="rasters_sam/", recursive=False)

    features = {obj.object_name for obj in objects if obj.object_name.endswith('/')}

    for feature in features:

        print(f"Merging rasters for feature: {feature}")

        feature = Path(feature).name

        if feature == "merged":
            continue

        local_paths = []
        for obj in minio_client.list_objects("pnoa-lidar", prefix=f"rasters_sam/{feature}/", recursive=True):
            local_path = Path(os.environ.get("TMP_DIR"), feature, Path(obj.object_name).name)
            minio_client.fget_object("pnoa-lidar", obj.object_name, local_path)
            local_paths.append(local_path)
        print("All data read")
        merged_path = merge_rasters(local_paths)

        minio_client.fput_object("pnoa-lidar", f"rasters_sam/merged/{feature}.tif", merged_path)

        if interpolating:
            interpolated_merged_path = Path(os.environ.get("TMP_DIR"), f"{feature}_interpolated.tif")
            subprocess.run(['gdal_fillnodata.py', '-md', '3', '-of', 'GTiff', merged_path, interpolated_merged_path])
            minio_client.fput_object("pnoa-lidar", f"rasters_sam/merged/{feature}_interpolated.tif", interpolated_merged_path)

            if crop:
                mask_path = Path(os.environ.get("TMP_DIR"), "mask.jp2")
                minio_client.fget_object("etc-products", "2021/June/S2A_MSIL2A_20210607T105621_N0300_R094_T30SUF_20210607T155717/raw/B02_10m.jp2", mask_path)
                cropped_path = Path(os.environ.get("TMP_DIR"), f"{feature}_interpolated_cropped.tif")
                reproject_and_crop_raster(interpolated_merged_path, mask_path, cropped_path)
                minio_client.fput_object("pnoa-lidar", f"rasters_sam/merged/{feature}_interpolated_cropped.tif", cropped_path)

            interpolated_merged_path.unlink()

        merged_path.unlink()

        print(f"Finished merging rasters for feature: {feature}")

if __name__ == "__main__":
    main(interpolating=True, crop=True)


