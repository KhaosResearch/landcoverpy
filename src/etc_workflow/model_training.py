import json
from os.path import join

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from etc_workflow.config import settings
from etc_workflow.confusion_matrix import compute_confusion_matrix
from etc_workflow.utils import _get_minio, _safe_minio_execute


def _feature_reduction(
    df_x: pd.DataFrame, df_y: pd.DataFrame, percentage_columns: int = 100
):
    """Feature reduction method. Receives the training dataset and returns a set of variables."""

    if percentage_columns < 100:
        n_columns = len(df_x.columns.tolist())
        n_features = int(n_columns * percentage_columns / 100)
        model = LogisticRegression(
            penalty="elasticnet", max_iter=10000, solver="saga", n_jobs=-1, l1_ratio=0.5
        )
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        fit = rfe.fit(df_x, df_y)
        used_columns = df_x.columns[fit.support_].tolist()
    else:
        used_columns = df_x.columns.tolist()

    return used_columns


def train_model_land_cover(land_cover_dataset: str, n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""

    training_dataset_path = join(settings.TMP_DIR, land_cover_dataset)

    minio_client = _get_minio()

    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, land_cover_dataset),
        file_path=training_dataset_path,
    )

    train_df = pd.read_csv(training_dataset_path)
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()

    minio_folder = settings.LAND_COVER_MODEL_FOLDER

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

    reduced_x_train_data = train_df[used_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_x_train_data, y_train_data, test_size=0.15
    )

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train.reindex(columns=used_columns)
    print(X_train)
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
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, minio_folder, confusion_image_filename),
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
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_name}",
        file_path=model_path,
        content_type="mlmodel/randomforest",
    )

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": n_jobs,
        "used_columns": list(used_columns),
        "classes": list(labels)
    }

    model_metadata_name = "metadata.json"
    model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

    with open(model_metadata_path, "w") as f:
        json.dump(model_metadata, f)

    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_metadata_name}",
        file_path=model_metadata_path,
        content_type="text/json",
    )

def train_model_forest(forest_dataset: str, use_open_forest: bool = False ,n_jobs: int = 2):
    """Trains a Random Forest model using a forest dataset."""

    training_dataset_path = join(settings.TMP_DIR, forest_dataset)

    minio_client = _get_minio()

    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, forest_dataset),
        file_path=training_dataset_path,
    )

    train_df = pd.read_csv(training_dataset_path)
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()

    minio_folder = ''
    if use_open_forest:
        train_df = train_df[train_df["class"] == "openForest"]
        minio_folder = settings.OPEN_FOREST_MODEL_FOLDER
    else:
        train_df = train_df[train_df["class"] != "openForest"]
        minio_folder = settings.DENSE_FOREST_MODEL_FOLDER

    y_train_data = train_df["forest_type"]
    x_train_data = train_df.drop(
        [
            "class",
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
            "forest_type",
        ],
        axis=1,
    )

    used_columns = _feature_reduction(x_train_data, y_train_data)

    reduced_x_train_data = train_df[used_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_x_train_data, y_train_data, test_size=0.30
    )

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train.reindex(columns=used_columns)
    print(X_train)
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
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, minio_folder, confusion_image_filename),
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
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_name}",
        file_path=model_path,
        content_type="mlmodel/randomforest",
    )

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": n_jobs,
        "used_columns": list(used_columns),
        "classes": list(labels)
    }
    model_metadata_name = "metadata.json"
    model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

    with open(model_metadata_path, "w") as f:
        json.dump(model_metadata, f)

    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_metadata_name}",
        file_path=model_metadata_path,
        content_type="text/json",
    )

