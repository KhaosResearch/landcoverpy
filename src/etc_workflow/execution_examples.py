from distributed import Client

from etc_workflow.config import settings
from etc_workflow.data_postprocessing import postprocess_dataset
from etc_workflow.model_training import train_model
from etc_workflow.plot_spectral_signature import compute_spectral_signature_plot
from etc_workflow.workflow import workflow


def run_compute_training_dataset_distributed():
    """Creates a training dataset in distributed mode"""

    client = Client(address=settings.DASK_CLUSTER_IP)
    workflow(predict=False, client=client)


def run_compute_training_dataset():
    """Creates a training dataset in local mode"""

    workflow(predict=False)


def run_model_training():
    """Postprocess the training dataset, trains a Random Forest using it and computes the spectral plot of classes `water`, `wetland`, `closedForest` and `shrubland`"""

    input_dataset = "dataset.csv"
    postprocessed_dataset = "dataset_postprocessed.csv"

    postprocess_dataset(input_dataset, postprocessed_dataset)
    train_model(postprocessed_dataset)
    compute_spectral_signature_plot(
        postprocessed_dataset,
        out_plot_path="/mnt/home/anmabur/etc-uma/etc-scripts/images/plot.png",
        classes_showed=["water", "wetland", "closedForest", "shrubland"],
    )


def run_predict_tiles_distributed():
    """Predicts all tiles appearing in the training dataset in distributed mode"""

    client = Client(address=settings.DASK_CLUSTER_IP)
    workflow(predict=True, client=client, tiles_to_predict=None)


def run_predict_tiles():
    """Predicts all tiles appearing in the training dataset in local mode"""

    workflow(predict=True, tiles_to_predict=None)
