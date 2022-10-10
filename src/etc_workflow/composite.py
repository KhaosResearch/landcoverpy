import traceback
from hashlib import sha256
from itertools import compress
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
from pymongo.collection import Collection
from scipy.ndimage import convolve

from etc_workflow import raw_index_calculation_composite
from etc_workflow.config import settings
from etc_workflow.execution_mode import ExecutionMode
from etc_workflow.minio import MinioConnection
from etc_workflow.utilities.raster import (
    _get_kwargs_raster,
    _get_product_rasters_paths,
    _get_raster_filename_from_path,
    _get_raster_name_from_path,
    _get_spatial_resolution_raster,
    _read_raster,
)
from etc_workflow.utilities.sentinel import _sentinel_date_to_datetime


def _composite(
    band_paths: List[str], method: str = "median", cloud_masks: List[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    Calculate the composite between a series of bands.

    Parameters:
        band_paths (List[str]) : List of paths to calculate the composite from.
        method (str) : To calculate the composite. Values: "median".
        cloud_masks (List[np.ndarray]) : Cloud masks of each band, cloudy pixels would not be taken into account for making the composite.

    Returns:
        (composite_out, composite_kwargs) (Tuple[np.ndarray, dict]) : Tuple containing the numpy array of the composed band, along with its kwargs.
    """
    composite_bands = []
    composite_kwargs = None
    for i in range(len(band_paths)):
        band_path = band_paths[i]
        if composite_kwargs is None:
            composite_kwargs = _get_kwargs_raster(band_path)

        if composite_kwargs["driver"] == "JP2OpenJPEG":
            composite_kwargs["dtype"] = "float32"
            composite_kwargs["nodata"] = np.nan
            composite_kwargs["driver"] = "GTiff"

        band = _read_raster(band_path)

        # Remove nodata pixels
        band = np.where(band == 0, np.nan, band)
        if i < len(cloud_masks):
            cloud_mask = cloud_masks[i]
            # Remove cloud-related pixels
            band = np.where(cloud_mask == 1, np.nan, band)

        composite_bands.append(band)
        Path.unlink(Path(band_path))

    shapes = [np.shape(band) for band in composite_bands]

    # Check if all arrays are of the same shape
    if not np.all(np.array(list(map(lambda x: x == shapes[0], shapes)))):
        raise ValueError(f"Not all bands have the same shape\n{shapes}")
    elif method == "median":
        composite_out = np.nanmedian(composite_bands, axis=0)
    else:
        raise ValueError(f"Method '{method}' is not recognized.")

    return (composite_out, composite_kwargs)


def _get_id_composite(products_ids: List[str], execution_mode: ExecutionMode) -> str:
    """
    Calculate the id of a composite using its products' ids and the execution mode.
    """
    products_ids.sort()
    mode_encoded = "1" if execution_mode != ExecutionMode.TRAINING else "0"
    concat_ids = "".join(products_ids)
    id_code = concat_ids + mode_encoded 
    hashed_ids = sha256(id_code.encode("utf-8")).hexdigest()
    return hashed_ids


def _get_title_composite(
    products_dates: List[str], products_tiles: List[str], composite_id: str, execution_mode: ExecutionMode
) -> str:
    """
    Get the title of a composite.
    If the execution mode is training, the title will contain the "S2E" prefix, else it will be "S2S".
    """
    if not all(product_tile == products_tiles[0] for product_tile in products_tiles):
        raise ValueError(
            f"Error creating composite, products have different tile: {products_tiles}"
        )
    tile = products_tiles[0]
    first_product_date, last_product_date = min(products_dates), max(products_dates)
    first_product_date = first_product_date.split("T")[0]
    last_product_date = last_product_date.split("T")[0]
    prefix = "S2S" if execution_mode != ExecutionMode.TRAINING else "S2E"
    composite_title = f"{prefix}_MSIL2A_{first_product_date}_NXXX_RXXX_{tile}_{last_product_date}_{composite_id[:8]}"
    return composite_title


# Intentar leer de mongo si existe algun composite con esos products
def _get_composite(
    products_metadata: Iterable[dict], mongo_collection: Collection, execution_mode: ExecutionMode
) -> dict:
    """
    Search a composite metadata in mongo.
    """
    products_ids = [products_metadata["id"] for products_metadata in products_metadata]
    hashed_id_composite = _get_id_composite(products_ids, execution_mode)
    composite_metadata = mongo_collection.find_one({"id": hashed_id_composite})
    return composite_metadata


def _create_composite(
    products_metadata: Iterable[dict],
    minio_client: MinioConnection,
    bucket_products: str,
    bucket_composites: str,
    mongo_composites_collection: Collection,
    execution_mode: ExecutionMode
) -> None:
    """
    Compose multiple Sentinel-2 products into a new product, this product is called "composite".
    Each band of the composite is computed using the pixel-wise median. Cloudy pixels of each product are not used in the median.
    Cloud masks are expanded if the execution mode is training, creating a composite with "S2E" prefix.
    If execution mode is predict, Sentinel's default cloud masks are used, creating a composite with "S2S" prefix.
    Once computed, the composite is stored in Minio, and its metadata in Mongo. 
    """

    products_titles = [product["title"] for product in products_metadata]
    print(
        "Creating composite of ", len(products_titles), " products: ", products_titles
    )

    products_ids = []
    products_titles = []
    products_dates = []
    products_tiles = []
    bands_paths_products = []
    cloud_masks_temp_paths = []
    cloud_masks = {"10": [], "20": [], "60": []}
    scl_cloud_values = [3, 8, 9, 10, 11]

    print("Downloading and reading the cloud masks needed to make the composite")
    for product_metadata in products_metadata:

        product_title = product_metadata["title"]
        products_ids.append(product_metadata["id"])
        products_titles.append(product_title)
        products_dates.append(product_title.split("_")[2])
        products_tiles.append(product_title.split("_")[5])

        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client, bucket_products
        )
        bands_paths_product = list(compress(rasters_paths, is_band))
        bands_paths_products.append(bands_paths_product)

        # Download cloud masks in all different spatial resolutions
        for band_path in bands_paths_product:
            band_name = _get_raster_name_from_path(band_path)
            band_filename = _get_raster_filename_from_path(band_path)
            if "SCL" in band_name:
                temp_dir_product = f"{settings.TMP_DIR}/{product_title}.SAFE"
                temp_path_product_band = f"{temp_dir_product}/{band_filename}"
                minio_client.fget_object(bucket_products, band_path, str(temp_path_product_band))
                cloud_masks_temp_paths.append(temp_path_product_band)
                spatial_resolution = str(
                    int(_get_spatial_resolution_raster(temp_path_product_band))
                )
                scl_band = _read_raster(temp_path_product_band)
                # Binarize scl band to get a cloud mask
                cloud_mask = np.isin(scl_band, scl_cloud_values).astype(np.bool)
                # Expand cloud mask for a more aggresive masking
                if execution_mode == ExecutionMode.TRAINING:
                    cloud_mask = _expand_cloud_mask(cloud_mask, int(spatial_resolution))
                cloud_masks[spatial_resolution].append(cloud_mask)
                kwargs = _get_kwargs_raster(temp_path_product_band)
                with rasterio.open(temp_path_product_band, "w", **kwargs) as f:
                    f.write(cloud_mask)

                # 10m spatial resolution cloud mask raster does not exists, have to be rescaled from 20m mask
                if "SCL_20m" in band_filename:
                    scl_band_10m_temp_path = temp_path_product_band.replace(
                        "_20m.jp2", "_10m.jp2"
                    )
                    cloud_mask_10m = _read_raster(
                        temp_path_product_band,
                        rescale=True,
                        path_to_disk=scl_band_10m_temp_path,
                        to_tif=False,
                    )
                    cloud_masks_temp_paths.append(scl_band_10m_temp_path)
                    cloud_masks["10"].append(cloud_mask_10m)

    composite_id = _get_id_composite(products_ids, execution_mode)
    composite_title = _get_title_composite(products_dates, products_tiles, composite_id, execution_mode)
    temp_path_composite = Path(settings.TMP_DIR, composite_title + ".SAFE")

    uploaded_composite_band_paths = []
    temp_paths_composite_bands = []
    temp_product_dirs = []
    result = None
    try:
        num_bands = len(bands_paths_products[0])
        for i_band in range(num_bands):
            products_i_band_path = [
                bands_paths_product[i_band]
                for bands_paths_product in bands_paths_products
            ]
            band_name = _get_raster_name_from_path(products_i_band_path[0])
            band_filename = _get_raster_filename_from_path(products_i_band_path[0])
            if "SCL" in band_name:
                continue
            temp_path_composite_band = Path(temp_path_composite, band_filename)

            temp_path_list = []

            for product_i_band_path in products_i_band_path:
                product_title = product_i_band_path.split("/")[2]

                temp_dir_product = f"{settings.TMP_DIR}/{product_title}.SAFE"
                temp_path_product_band = f"{temp_dir_product}/{band_filename}"
                print(
                    f"Downloading raster {band_name} from minio into {temp_path_product_band}"
                )
                minio_client.fget_object(bucket_products, product_i_band_path, str(temp_path_product_band))
                spatial_resolution = str(
                    int(_get_spatial_resolution_raster(temp_path_product_band))
                )

                if temp_dir_product not in temp_product_dirs:
                    temp_product_dirs.append(temp_dir_product)
                temp_path_list.append(temp_path_product_band)
            composite_i_band, kwargs_composite = _composite(
                temp_path_list,
                method="median",
                cloud_masks=cloud_masks[spatial_resolution],
            )

            # Save raster to disk
            if not Path.is_dir(temp_path_composite):
                Path.mkdir(temp_path_composite)

            temp_path_composite_band = str(temp_path_composite_band)
            if temp_path_composite_band.endswith(".jp2"):
                temp_path_composite_band = temp_path_composite_band[:-3] + "tif"

            temp_path_composite_band = Path(temp_path_composite_band)

            temp_paths_composite_bands.append(temp_path_composite_band)

            with rasterio.open(
                temp_path_composite_band, "w", **kwargs_composite
            ) as file_composite:
                file_composite.write(composite_i_band)

            # Upload raster to minio
            band_filename = band_filename[:-3] + "tif"
            minio_band_path = f"{composite_title}/raw/{band_filename}"
            minio_client.fput_object(
                bucket_name=bucket_composites,
                object_name=minio_band_path,
                file_path=temp_path_composite_band,
                content_type="image/tif",
            )
            uploaded_composite_band_paths.append(minio_band_path)
            print(
                f"Uploaded raster: -> {temp_path_composite_band} into {bucket_composites}:{minio_band_path}"
            )

        composite_metadata = dict()
        composite_metadata["id"] = composite_id
        composite_metadata["title"] = composite_title
        composite_metadata["products"] = [
            dict(id=product_id, title=products_title)
            for (product_id, products_title) in zip(products_ids, products_titles)
        ]
        composite_metadata["first_date"] = _sentinel_date_to_datetime(
            min(products_dates)
        )
        composite_metadata["last_date"] = _sentinel_date_to_datetime(
            max(products_dates)
        )

        # Upload metadata to mongo
        result = mongo_composites_collection.insert_one(composite_metadata)
        print("Inserted data in mongo, id: ", result.inserted_id)

        # Compute indexes
        raw_index_calculation_composite.calculate_raw_indexes(
            _get_id_composite(
                [products_metadata["id"] for products_metadata in products_metadata], execution_mode
            )
        )

    except (Exception, KeyboardInterrupt) as e:
        print("Removing uncompleted composite from minio")
        traceback.print_exc()
        for composite_band in uploaded_composite_band_paths:
            minio_client.remove_object(
                bucket_name=bucket_composites, object_name=composite_band
            )
        products_ids = [
            products_metadata["id"] for products_metadata in products_metadata
        ]
        mongo_composites_collection.delete_one({"id": _get_id_composite(products_ids)})
        raise e

    finally:
        for composite_band in temp_paths_composite_bands + cloud_masks_temp_paths:
            Path.unlink(Path(composite_band))


def _expand_cloud_mask(cloud_mask: np.ndarray, spatial_resolution: int):
    """
    Empirically-tested method for expanding a cloud mask using convolutions.
    Method specifically developed for expanding the Sentinel-2 cloud mask (values of SCL), as provided cloud mask is very conservative.
    This method reduces the percentage of false negatives around true positives, but increases the percentage os false positives.
    We fear false negatives more than false positives, because false positives will usually disappear when a composite is done.

    Parameters:
        cloud_mask (np.ndarray) : Boolean array, ones are cloudy pixels and zeros non-cloudy pixels.
        spatial_resolution (int) : Spatial resolution of the image.
                                   It has to be provided because the convolution kernel should cover a 600 x 600 m2 area,
                                   which is the ideal for expanding the Sentinel-2 cloud mask.
    Returns:
        expanded_mask (np.ndarray) : Expanded cloud mask.
    """
    kernel_side_meters = 600  # The ideal kernel area 600m x 600m. Empirically tested.
    kernel_side = int(kernel_side_meters / spatial_resolution)
    v_kernel = np.ones(
        (kernel_side, 1)
    )  # Apply convolution separability property to reduce computation time
    h_kernel = np.ones((1, kernel_side))
    cloud_mask = cloud_mask.astype(np.float32)
    convolved_mask_v = convolve(
        cloud_mask[0], v_kernel, mode="reflect"
    )  # Input matrices has to be 2-D
    convolved_mask = convolve(convolved_mask_v, h_kernel, mode="reflect")
    convolved_mask = convolved_mask[np.newaxis, :, :]
    expanded_mask = np.where(
        convolved_mask >= (kernel_side * kernel_side) * 0.075, 1, cloud_mask
    )
    return expanded_mask
