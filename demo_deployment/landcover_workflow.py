import os
import json
from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.workflow import workflow
from landcoverpy.model_training import train_model_land_cover, train_second_level_models
from landcoverpy.utilities.geometries import _group_validated_data_points_by_tile, _kmz_to_geojson, _csv_to_geojson


def landcover_workflow():

    # 1. Generate the training data
    workflow(execution_mode=ExecutionMode.TRAINING, use_aster=False)


    # 2. Train the models 
    train_model_land_cover()
    
    subcategories_to_predict_env = os.getenv("SUBCATEGORY_PREDICTION", "[]")
    try:
        subcategories_to_predict = set(json.loads(subcategories_to_predict_env))
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing SUBCATEGORY_PREDICTION: {subcategories_to_predict_env}. Expected format: '['category1', 'category2', ...]'")
    
    if len(subcategories_to_predict) != 0:
        train_second_level_models(subcategories_to_predict)


    # 3. Get tiles to predict
    if os.getenv("TILES_TO_PREDICT").lower() == "prediction":
        data_file = os.getenv("DB_FILE")
        if data_file.endswith(".kmz"):
            data_file = _kmz_to_geojson(data_file)
        if data_file.endswith(".csv"):
            data_file = _csv_to_geojson(data_file, sep=';')
        polygons_per_tile = _group_validated_data_points_by_tile(data_file)
        tiles_to_predict = list(polygons_per_tile.keys())
    else:
        tiles_to_predict_env = os.getenv("TILES_TO_PREDICT", "[]")
        try:
            tiles_to_predict = set(json.loads(tiles_to_predict_env))
        except json.JSONDecodeError:
            raise ValueError(f"Error parsing TILES_TO_PREDICT: {tiles_to_predict_env}. Expected format: '['NNLLL', 'NNLLL', ...]'")
    print("Tiles to predict: ", tiles_to_predict)


    # 4. Generate prediction maps
    slices = (5,5)
    workflow(execution_mode=ExecutionMode.LAND_COVER_PREDICTION, tiles_to_predict=tiles_to_predict, window_slices=slices, use_aster=False)
    if len(subcategories_to_predict) != 0:
        workflow(ExecutionMode.SECOND_LEVEL_PREDICTION, tiles_to_predict=tiles_to_predict, window_slices=slices, use_aster=False)