from os.path import join
from typing import List

from etc_workflow.config import settings
from etc_workflow.utils import _get_minio, _safe_minio_execute

def download_classified_tiles(downloaded_tiles: List[str], destination_folder: str):


    minio_client = _get_minio()

    # List all classifications in minio
    classified_tiles_cursor = minio_client.list_objects(
        bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
        prefix=join(settings.MINIO_DATA_FOLDER_NAME, "forest_classification")
    )

    # For each image
    for classified_tile_object in classified_tiles_cursor:
        classification_path = classified_tile_object.object_name
        
        classification_filename = classification_path.split("/")[-1]
        classification_tile = classification_filename.split("_")[-1].split(".")[0]
        
        if classification_tile in downloaded_tiles:
            print(classification_tile)
            local_classification_path = join(
                destination_folder, classification_filename
            )

            _safe_minio_execute(
                func=minio_client.fget_object,
                bucket_name=settings.MINIO_BUCKET_CLASSIFICATIONS,
                object_name=classification_path,
                file_path=local_classification_path
            )