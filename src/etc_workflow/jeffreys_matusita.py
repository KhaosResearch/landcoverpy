from os.path import join

from itertools import combinations
import numpy as np
from numpy.lib.function_base import cov
import pandas as pd

from etc_workflow.config import settings
from etc_workflow.utils import _get_minio, _safe_minio_execute

def _jeffreys_matusita_distance(mu1, sigma1, mu2, sigma2):
    jmd = np.sqrt(2*(1-np.exp(-_bhattacharyya_distance(mu1, sigma1, mu2, sigma2))))
    return jmd*np.sqrt(2)

def _bhattacharyya_distance(mu1, sigma1, mu2, sigma2):

    sigma = np.mean([sigma1,sigma2], axis=0)
    
    ds = np.linalg.slogdet(sigma)[1]
    ds1 = np.linalg.slogdet(sigma1)[1]
    ds2 = np.linalg.slogdet(sigma2)[1]
    bd = (_squared_mahalanobis_distance(mu1, mu2, sigma)/8) + (0.5) * ((ds - ds1/2 - ds2/2)/2)
    return bd


def _squared_mahalanobis_distance(u, v, sigma):
    delta = u - v
    try:
        inv_sigma = np.linalg.inv(sigma)
    except:
        inv_sigma = np.linalg.pinv(sigma)
    m = (delta @ inv_sigma) @ delta
    return m

def jeffreys_matusita_analysis(
    input_dataset: str,
    out_path: str,
    is_forest: bool = False,
):

    dataset_path = join(settings.TMP_DIR, input_dataset)

    minio_client = _get_minio()

    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, input_dataset),
        file_path=dataset_path,
    )

    df = pd.read_csv(dataset_path)
    df = df.drop(
        [
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
            "summer_cri1",
            "autumn_cri1",
            "spring_cri1",
        ],
        axis=1,
    )
    
    if is_forest:
        class_column = "forest_type"
        df = df.drop("class",axis=1)
    else:
        class_column = "class"

    df_byclass = df.groupby(by=class_column)
    means = df_byclass.mean()
    covs = df_byclass.cov()
    classes = df[class_column].unique()
    with open(out_path, mode='w') as outfile:
        outfile.write("CLASS1;CLASS2;DISTANCE\n")
        for (class1, class2) in combinations(classes, r=2):
            mu1, mu2 = means.loc[class1], means.loc[class2]
            sigma1, sigma2 = covs.loc[class1], covs.loc[class2]
            m = _jeffreys_matusita_distance(mu1, sigma1, mu2, sigma2)
            outfile.write(f"{class1};{class2};{m}\n")

