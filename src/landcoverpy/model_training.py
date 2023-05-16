
from os.path import join
import time
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import settings
from utilities.confusion_matrix import compute_confusion_matrix


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


def get_country(lat, lon):
        # Crear un objeto geolocalizador
        geolocator = Nominatim(user_agent="my_geocoder")
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
        if location is not None:
            return location.raw['address'].get('country',  '')
        else:
            return None
        

def save_df_with_countries(df):
    df["country"] = df.apply(lambda i: get_country(i['latitude'], i['longitude']), axis=1)
    time.sleep(1)
    return df

def train_model_land_cover(n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""


    train_df = pd.read_csv("/home/irene/Working_dir/landcoverpy/data/dataset_postprocessed.csv")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()

    train_df_head = train_df.head(20).copy()

    df_with_country = save_df_with_countries(train_df_head)
    
    # Aplicar la función a las columnas de latitud y longitud en el dataframe
    #train_df["country"] = train_df.apply(lambda i: get_country(i['latitude'], i['longitude']), axis=1)
    # train_df_head["country"] = train_df_head.apply(lambda i: get_country(i['latitude'], i['longitude']), axis=1)
    # time.sleep(1)



    train_grouped = df_with_country.groupby("country").apply(lambda x: x.reset_index(drop=True))


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

    print("X_test",X_test)

    labels = y_train_data.unique()
    for country, df_test_country in X_test.groupby("country"):
        # Obtén las características (X) y las etiquetas (y) para el país actual
        X_country = df_test_country
        y_country = y_test[df_test_country.index]
        y_country1 = y_test[country]

       

        print("ffffff",y_country)
        #print("jjj", y_country)
        
        # Realiza las predicciones para el país actual utilizando tu modelo entrenado
        y_pred_pais = clf.predict(X_country)
        print("SDFG", y_pred_pais)
        
     

        confusion_image_filename = f"/home/irene/Working_dir/landcoverpy/confusion_matrix_{country}.png"

        #out_image_path = join(settings.TMP_DIR, confusion_image_filename)
        compute_confusion_matrix(y_pred_pais, y_country, labels, out_image_path=confusion_image_filename)
        


    # for pais in unique_countries:
    #     Filtrar el conjunto de prueba solo para el país actual
    #     X_pais = X_test[y_test == pais]
    #     y_pais = y_test[y_test == pais]
    #     print(X_pais)

    #     Predecir las etiquetas para el país actual
    #     y_pred_pais = clf.predict(X_pais)


    #     confusion_image_filename = f"confusion_matrix_{pais}.png"

    #     out_image_path = join(settings.TMP_DIR, confusion_image_filename)
    #     compute_confusion_matrix(y_pred_pais, y_test, labels, out_image_path=out_image_path)






train_model_land_cover(2)