from distributed import Client

from etc_workflow.config import settings
from etc_workflow.data_postprocessing import postprocess_dataset
from etc_workflow.model_training import train_model
from etc_workflow.workflow import workflow
from etc_workflow.plot_spectral_signature import compute_spectral_signature_plot



def run_compute_training_dataset_distributed():
    """Creates a training dataset in distributed mode"""

    client = Client(address=settings.DASK_CLUSTER_IP)
    workflow(predict=False, client=client)


def run_compute_training_dataset():
    """Creates a training dataset in local mode"""

    workflow(predict=False)

def run_model_training():
    """Postprocess the training dataset, trains a Random Forest using it and computes the spectral plot of classes `water`, `wetland`, `bosque` and `matorral`"""

    input_dataset = "dataset.csv"
    postprocessed_dataset = "dataset_postprocessed.csv"

    postprocess_dataset(input_dataset, postprocessed_dataset)
    train_model(postprocessed_dataset)
    compute_spectral_signature_plot(
        postprocessed_dataset, 
        out_plot_path="/mnt/home/anmabur/etc-uma/etc-scripts/images/plot.png", 
        classes_showed=["water", "wetland", "bosque", "matorral"]
    )

def run_predict_tiles_distributed():
    """Predicts all tiles appearing in the training dataset in distributed mode"""

    client = Client(address=settings.DASK_CLUSTER_IP)
    columns_predict = sorted(["slope","aspect","dem","spring_cri1","spring_ri","spring_evi2","spring_mndwi","spring_moisture","spring_ndyi","spring_ndre","spring_ndvi","spring_osavi","spring_AOT","spring_B01","spring_B02","spring_B03","spring_B04","spring_B05","spring_B06","spring_B07","spring_B08","spring_B09","spring_B11","spring_B12","spring_B8A","summer_cri1","summer_ri","summer_evi2","summer_mndwi","summer_moisture","summer_ndyi","summer_ndre","summer_ndvi","summer_osavi","summer_AOT","summer_B01","summer_B02","summer_B03","summer_B04","summer_B05","summer_B06","summer_B07","summer_B08","summer_B09","summer_B11","summer_B12","summer_B8A","autumn_cri1","autumn_ri","autumn_evi2","autumn_mndwi","autumn_moisture","autumn_ndyi","autumn_ndre","autumn_ndvi","autumn_osavi","autumn_AOT","autumn_B01","autumn_B02","autumn_B03","autumn_B04","autumn_B05","autumn_B06","autumn_B07","autumn_B08","autumn_B09", "autumn_B11", "autumn_B12", "autumn_B8A"])

    workflow(predict=True, client=client, tiles_to_predict=None, columns_predict=columns_predict)

def run_predict_tiles():
    """Predicts all tiles appearing in the training dataset in local mode"""
    columns_predict = sorted(["slope","aspect","dem","spring_cri1","spring_ri","spring_evi2","spring_mndwi","spring_moisture","spring_ndyi","spring_ndre","spring_ndvi","spring_osavi","spring_AOT","spring_B01","spring_B02","spring_B03","spring_B04","spring_B05","spring_B06","spring_B07","spring_B08","spring_B09","spring_B11","spring_B12","spring_B8A","summer_cri1","summer_ri","summer_evi2","summer_mndwi","summer_moisture","summer_ndyi","summer_ndre","summer_ndvi","summer_osavi","summer_AOT","summer_B01","summer_B02","summer_B03","summer_B04","summer_B05","summer_B06","summer_B07","summer_B08","summer_B09","summer_B11","summer_B12","summer_B8A","autumn_cri1","autumn_ri","autumn_evi2","autumn_mndwi","autumn_moisture","autumn_ndyi","autumn_ndre","autumn_ndvi","autumn_osavi","autumn_AOT","autumn_B01","autumn_B02","autumn_B03","autumn_B04","autumn_B05","autumn_B06","autumn_B07","autumn_B08","autumn_B09", "autumn_B11", "autumn_B12", "autumn_B8A"])

    workflow(predict=True, tiles_to_predict=None, columns_predict=columns_predict)
