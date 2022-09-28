from os.path import join

from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.function_base import cov
import pandas as pd
import seaborn as sns

from etc_workflow.config import settings
from etc_workflow.utilities.utils import _get_minio, _safe_minio_execute

def _jeffreys_matusita_distance(mu1, sigma1, mu2, sigma2):
    """
    Computes the Jeffreys-Matusita distance between two distributions `P` and `Q`.
    Parameters:
        mu1 (np.ndarray) : Mean vector of `P`.
        sigma1 (np.ndarray) : Covariance matrix of `P`.
        mu2 (np.ndarray) : Mean vector of `Q`.
        sigma2 (np.ndarray) : Covariance matrix of `Q`.

    Returns:
        jmd (int) : Jeffreys-Matusita distanc.

    """
    jmd = np.sqrt(2*(1-np.exp(-_bhattacharyya_distance(mu1, sigma1, mu2, sigma2))))
    return jmd

def _bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    """
    Computes the Bhattarcharyya distance between two distributions `P` and `Q`.
    Parameters:
        mu1 (np.ndarray) : Mean vector of `P`.
        sigma1 (np.ndarray) : Covariance matrix of `P`.
        mu2 (np.ndarray) : Mean vector of `Q`.
        sigma2 (np.ndarray) : Covariance matrix of `Q`.

    Returns:
        db (int) : Bhattarcharyya distance.

    """
    sigma = np.mean([sigma1,sigma2], axis=0)
    
    ds = np.linalg.slogdet(sigma)[1]
    ds1 = np.linalg.slogdet(sigma1)[1]
    ds2 = np.linalg.slogdet(sigma2)[1]
    bd = (_squared_mahalanobis_distance(mu1, mu2, sigma)/8) + (0.5) * ((ds - ds1/2 - ds2/2)/2)
    return bd


def _squared_mahalanobis_distance(u, v, sigma):
    """
    Computes the squared Mahalanobis distance between a point and a distribution `Q`.
    Parameters:
        u (np.ndarray) : A point located in the same dimension space than `Q`
        v (np.ndarray) : Mean vector of `Q`.
        value2 (np.ndarray) : Covariance matrix of `Q`.

    Returns:
        m (int) : Squared Mahalanobis distance between a point `u` to a distribution defined by mean vector `v` and covariance matrix `sigma`.

    """
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
    """
    Computes the Jeffreys-Matusita  distance for all class combinations in `input_Dataset`. The distance is multiplied by sqrt(2) in order to map the distance to 0-2.
    If J-M distance is lower than 1.9 between two different classes, it is considered that those classes are not separable with the current attributes.
    Parameters:
        input_dataset (str) : Name of the dataset stored in MinIO.
        out_path (str) : Local folder where a csv file and a plot will be stored.
        is_forest (bool) : Indicate if classes used are those in land cover or in forest classification.
    """

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
    data = []
    data_complete = []
    for (class1, class2) in combinations(classes, r=2):
        mu1, mu2 = means.loc[class1], means.loc[class2]
        sigma1, sigma2 = covs.loc[class1], covs.loc[class2]
        m = _jeffreys_matusita_distance(mu1, sigma1, mu2, sigma2) * np.sqrt(2)
        data.append([class1, class2, m])
        data_complete.append([class1, class2, m])
        data_complete.append([class2, class1, m])
    for class_ in classes:
        data_complete.append([class_,class_,2.0])
    matusita_df = pd.DataFrame(data, columns=['CLASS1', 'CLASS2', 'DISTANCE'])
    matusita_df.sort_values(by=["DISTANCE"], ascending=True)
    matusita_df.to_csv(join(out_path,"matusita.csv"), index=False)
    matusita_df_complete = pd.DataFrame(data_complete, columns=['CLASS1', 'CLASS2', 'DISTANCE'])
    matusita_df_complete = matusita_df_complete.pivot('CLASS1', 'CLASS2', 'DISTANCE')
    sns.heatmap(matusita_df_complete, vmin=1.8, vmax=2, center=1.95, cmap="RdYlGn", square=True, cbar=False)
    plt.savefig(join(out_path,"matusita.png"), bbox_inches="tight")

