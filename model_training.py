import pandas as pd
import numpy as np
from utils import pca
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils import get_minio, connect_mongo_composites_collection,get_product_rasters_paths, get_raster_name_from_path
from config import settings
from itertools import compress


# Temporal code to load premade dataset for training 


training = True
pc_columns = ['aspect', 'autumn_B01', 'autumn_evi', 'spring_AOT', 'spring_B01', 'spring_WVP', 'spring_evi', 'summer_B01', 'summer_B02', 'summer_evi', 'summer_moisture']


if training:
    train_df = pd.read_csv("dataset_pca_full.csv")
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()
    # Prepare data for training
    #x_train_data = train_df.drop("class", axis=1)
    x_train_data = train_df.drop(["spring_SCL","autumn_SCL","summer_SCL"], axis=1)
    pc_columns = pca(x_train_data,95)
    print(pc_columns)

    train_df = pd.read_csv("dataset.csv")
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()
    y_train_data = train_df["class"] 
    reduced_x_train_data = train_df[pc_columns]
    y_train_data = y_train_data.mask(y_train_data != 'unclassified', 'vegetal')
    # Filter bands according to PCA, matusita etc
    reduced_x_train_data.to_csv("dataset_pca.csv",index=False)
    y_train_data.to_csv("ground_truth_pca.csv",index=False)
    # Train model 
    clf = RandomForestClassifier()
    clf.fit(reduced_x_train_data, y_train_data)
    joblib.dump(clf, 'model.pkl', compress=1)
else: # Prediction

    minio_client = get_minio()
    mongo_collection = connect_mongo_composites_collection()
    current_bucket = settings.MINIO_BUCKET_NAME_COMPOSITES
    product_metadata = mongo_collection.find_one({
                    "title": "S2X_MSIL2A_20200304_NXXX_RXXX_T30SUF_20200329_c07b4cb5"
                })
    (rasters_paths, is_band) = get_product_rasters_paths(product_metadata, minio_client, current_bucket)
    pc_raster_paths = []
    for pc_column in pc_columns:
        print(pc_column)
        for raster_path in rasters_paths:
            raster_name = get_raster_name_from_path(raster_path)
            if pc_column in raster_name:
                pc_raster_paths.append(raster_path)
    print(pc_raster_paths)

    #test_df = pd.read_csv("dataset_pca_full.csv")
    #test_df = test_df.fillna(np.nan)
    #test_df = test_df.dropna()
    #reduced_predict_data = test_df[pc_columns] # This should be coming from the PCA used during training
    #clf = joblib.load('model.pkl')
    #results = clf.predict(reduced_predict_data)
    #results.to_csv("predictions.csv", index=False)