import pandas as pd
from os.path import join

from etc_workflow.config import settings
from etc_workflow.utils import _get_minio, _safe_minio_execute


def postprocess_dataset(input_dataset: str, output_dataset: str):
    """
    Once the training dataset is computed, it is possible that it needs some preprocessing.
    This script is in charge of all kind of preprocessing needed. Right now, the steps are:
        - Read input dataset `input_dataset` from minio
        - Map classes `mixedBosque`, `plantacion`, and `bosqueRibera` to `bosque`
        - Map class `pastos` to `dehesas`
        - Reduce the quantity of rows labeled as `bosque` in a 1:9 ratio.
        - Shuffle the whole dataset
        - Write resulting dataset to Minio as `output_dataset`
    """
    minio_client = _get_minio()

    input_file_path = join(settings.TMP_DIR, input_dataset)

    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, input_dataset),
        file_path=input_file_path
    )

    df = pd.read_csv(input_file_path)
    print("Histogram of classes in input dataset\n", df["class"].value_counts())

    # Sustituir clases
    df["class"].replace(
        to_replace=["mixedBosque", "plantacion", "bosqueRibera"],
        value="bosque",
        inplace=True,
    )
    df["class"].replace(to_replace="pastos", value="dehesas", inplace=True)

    # Reducir tamaño datos bosque
    df_bosque = df[df["class"] == "bosque"]
    df_not_bosque = df[df["class"] != "bosque"]
    df_bosque = df_bosque.drop_duplicates(subset=["longitude", "latitude"]).reset_index(
        drop=True
    )
    df = pd.concat([df_bosque, df_not_bosque])

    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    print("Histogram of classes in output dataset\n", df["class"].value_counts())

    output_file_path = join(settings.TMP_DIR, output_dataset)

    df.to_csv(output_file_path, index=False)

    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, output_dataset),
        file_path=output_file_path,
        content_type="text/csv",
    )
