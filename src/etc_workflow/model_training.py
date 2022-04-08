import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from os.path import join

from etc_workflow.config import settings
from etc_workflow.confusion_matrix import compute_confusion_matrix
from etc_workflow.utils import _get_minio, _safe_minio_execute

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def _feature_reduction(df_x: pd.DataFrame, df_y: pd.DataFrame):
    """Feature reduction method. Receives the training dataset and returns a set of variables."""

    model = LogisticRegression(penalty="elasticnet", max_iter=1000, solver="saga", n_jobs=-1, l1_ratio=0.5)
    rfe = RFE(estimator=model, n_features_to_select=30)
    fit = rfe.fit(df_x, df_y)

    used_columns = sorted(
        [
            "slope",
            "aspect",
            "dem",
            "spring_cri1",
            "spring_ri",
            "spring_evi2",
            "spring_mndwi",
            "spring_moisture",
            "spring_ndyi",
            "spring_ndre",
            "spring_ndvi",
            "spring_osavi",
            "spring_AOT",
            "spring_B01",
            "spring_B02",
            "spring_B03",
            "spring_B04",
            "spring_B05",
            "spring_B06",
            "spring_B07",
            "spring_B08",
            "spring_B09",
            "spring_B11",
            "spring_B12",
            "spring_B8A",
            "summer_cri1",
            "summer_ri",
            "summer_evi2",
            "summer_mndwi",
            "summer_moisture",
            "summer_ndyi",
            "summer_ndre",
            "summer_ndvi",
            "summer_osavi",
            "summer_AOT",
            "summer_B01",
            "summer_B02",
            "summer_B03",
            "summer_B04",
            "summer_B05",
            "summer_B06",
            "summer_B07",
            "summer_B08",
            "summer_B09",
            "summer_B11",
            "summer_B12",
            "summer_B8A",
            "autumn_cri1",
            "autumn_ri",
            "autumn_evi2",
            "autumn_mndwi",
            "autumn_moisture",
            "autumn_ndyi",
            "autumn_ndre",
            "autumn_ndvi",
            "autumn_osavi",
            "autumn_AOT",
            "autumn_B01",
            "autumn_B02",
            "autumn_B03",
            "autumn_B04",
            "autumn_B05",
            "autumn_B06",
            "autumn_B07",
            "autumn_B08",
            "autumn_B09",
            "autumn_B11",
            "autumn_B12",
            "autumn_B8A",
        ]
    )

    used_columns = sorted(df_x.columns[fit.support_].tolist())

    return used_columns


def train_model(input_training_dataset: str, n_jobs: int = 2):
    """Trains a Random Forest model using a dataset."""

    training_dataset_path = join(settings.TMP_DIR, input_training_dataset)

    minio_client = _get_minio()

    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, input_training_dataset),
        file_path=training_dataset_path
    )

    train_df = pd.read_csv(training_dataset_path)
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()

    y_train_data = train_df["class"]
    x_train_data = train_df.drop(
        [
            "class",
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
        ],
        axis=1,
    )

    used_columns = _feature_reduction(x_train_data, y_train_data)
    print(used_columns)

    reduced_x_train_data = train_df[used_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_x_train_data, y_train_data, test_size=0.15
    )

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    y_true = clf.predict(X_test)

    labels = y_train_data.unique()

    confusion_image_filename = "confusion_matrix.png"
    out_image_path = join(settings.TMP_DIR, confusion_image_filename)
    compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path)

    # Save confusion matrix image to minio
    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME,confusion_image_filename),
        file_path=out_image_path,
        content_type="image/png",
    )

    model_name = "model.joblib"
    model_path = join(settings.TMP_DIR, model_name)
    joblib.dump(clf, model_path)

    # Save model to minio
    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{model_name}",
        file_path=model_path,
        content_type="mlmodel/randomforest",
    )

    model_metadata = {"model": str(type(clf)), "n_jobs": n_jobs, "used_columns": used_columns}
    model_metadata_name = 'metadata.json'
    model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

    with open(model_metadata_path, 'w') as f:
        json.dump(model_metadata, f)

    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{model_metadata_name}",
        file_path=model_metadata_path,
        content_type="text/json",
    )

train_model("dataset_postprocessed.csv", n_jobs=1)