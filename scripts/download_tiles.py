from os.path import join
from typing import List

from landcoverpy.config import settings
from landcoverpy.minio import MinioConnection

import rasterio
from rasterio.merge import merge

def download_classified_tiles(downloaded_tiles: List[str], destination_folder: str):

    src_files_to_mosaic = []

    minio_client = MinioConnection()
    # List all classifications in minio
    classified_tiles_cursor = minio_client.list_objects(
        bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
        prefix=join(settings.MINIO_DATA_FOLDER_NAME,"forest-0-6", "200m", "")
    )
    # For each image
    for classified_tile_object in classified_tiles_cursor:
        classification_path = classified_tile_object.object_name
        print(classification_path)
        
        classification_filename = classification_path.split("/")[-1]
        classification_tile = classification_filename.split("_")[-1].split(".")[0]
        
        if classification_tile in downloaded_tiles:
            print(classification_tile)

            local_classification_path = join(
                destination_folder, classification_filename
            )

            minio_client.fget_object(
                bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
                object_name=classification_path,
                file_path=local_classification_path
            )

            src = rasterio.open(local_classification_path)
            src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    # Obtener los metadatos de uno de los archivos de entrada
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # Guardar el archivo fusionado
    with rasterio.open(join(
                destination_folder, "merged_forest_spain.tif"
            ), "w", **out_meta) as dest:
        dest.write(mosaic)


from landcoverpy.utilities.aoi_tiles import get_list_of_tiles_in_iberian_peninsula
download_classified_tiles(get_list_of_tiles_in_iberian_peninsula(),"merged")