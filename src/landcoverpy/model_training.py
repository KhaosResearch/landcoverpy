
from os.path import join
import time
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import geocoder
from config import settings
from utilities.confusion_matrix import compute_confusion_matrix
import geopandas as gpd
from shapely.geometry import Point
import os 


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





def get_country(geodf, row):
    latitude = row['latitude']
    longitude = row['longitude']

    for index, feature in geodf.iterrows():
        if feature['geometry'].contains(Point(longitude, latitude)):
            return feature['name']
    
    return None


def get_country1(lat, lon):
        # Crear un objeto geolocalizador
        geolocator = Nominatim(user_agent="my_geocoder")
        
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
        if location is not None:
            return location.raw['address'].get('country',  '')
        else:
            return None
        

def train_model_land_cover(n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""


    train_df = pd.read_csv("./data/dataset_postprocessed.csv")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(np.nan)
    train_df= train_df.dropna()

    gdf = gpd.read_file('./geojson/global.json')

    train_df['country'] = train_df.apply(lambda row: get_country(gdf, row), axis=1)
    
    
    train_grouped = train_df.groupby("country").apply(lambda x: x.reset_index(drop=True))

    y_train_data = train_grouped["class"]
    x_train_data = train_grouped.drop(
        [
            "class",
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
            "country"
        ],
        axis=1,
    )


    used_columns = _feature_reduction(x_train_data, y_train_data)

    reduced_x_train_data = train_grouped[used_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_x_train_data, y_train_data, test_size=0.15
    )

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train.reindex(columns=used_columns)
    
    clf.fit(X_train, y_train)


    labels = y_train_data.unique()
    for country, df_test_country in X_test.groupby("country"):
        # Obtén las características (X) y las etiquetas (y) para el país actual
        X_country = df_test_country
        y_country = y_test[df_test_country.index]

        
        # Realiza las predicciones para el país actual utilizando tu modelo entrenado
        y_pred_pais = clf.predict(X_country)

        confusion_image_filename = f"./img_land/confusion_matrix_{country}.png"

        compute_confusion_matrix(y_pred_pais, y_country, labels, country, out_image_path=confusion_image_filename)
        

def train_model_forest(use_open_forest: bool = False ,n_jobs: int = 2):
    """Trains a Random Forest model using a forest dataset."""


    train_df = pd.read_csv("./data/dataset_forests.csv")

    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()
    
    #train_df = train_df.head(20).copy()
    
    
    if use_open_forest:
        print("si")

        train_df = train_df[train_df["class"] == "openForest"]
        equivalents = {4: 'Conifers', 5: 'Leafy', 6: 'Mixed', 0: 'Others'}
        train_df['forest_type'] = train_df['forest_type'].replace(equivalents)
    else:
        print("no")
        train_df = train_df[train_df["class"] != "openForest"]
        equivalents = {1: 'Conifers', 2: 'Leafy', 3: 'Mixed', 0: 'Other'}
        train_df['forest_type'] = train_df['forest_type'].replace(equivalents)
        
    gdf = gpd.read_file('./geojson/global.json')
    
    #train_df['country'] = train_df.apply(lambda i: get_country1(i['latitude'], i['longitude']), axis=1)
    train_df['country'] = train_df.apply(lambda row: get_country(gdf, row), axis=1)
    train_grouped = train_df.groupby("country").apply(lambda x: x.reset_index(drop=True))
    
    y_train_data = train_grouped["forest_type"]
    x_train_data = train_grouped.drop(
        [
            "class",
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
            "forest_type",
            "country"
        ],
        axis=1,
    )

    used_columns = _feature_reduction(x_train_data, y_train_data)

    reduced_x_train_data = train_grouped[used_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_x_train_data, y_train_data, test_size=0.15
    )

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train.reindex(columns=used_columns)
    
    clf.fit(X_train, y_train)

    labels = y_train_data.unique()
    
    print(X_train)
    for country, df_test_country in X_test.groupby("country"):
            # Obtén las características (X) y las etiquetas (y) para el país actual
            X_country = df_test_country
            y_country = y_test[df_test_country.index]

            
            # Realiza las predicciones para el país actual utilizando tu modelo entrenado
            y_pred_pais = clf.predict(X_country)

            confusion_image_filename = f"./img_forest/confusion_matrix_{country}.png"

            compute_confusion_matrix(y_pred_pais, y_country, labels, country, out_image_path=confusion_image_filename)

train_model_land_cover(2)

#train_model_forest(True, 2)

