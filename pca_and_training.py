from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
from utils import normalize, pca
import joblib
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix
from visualize_confusion_matrix import visualize_confusion_matrix


# Temporal code to load premade dataset for training 


training = True

train_df = pd.read_csv("dataset.csv")
train_df = train_df.replace([np.inf, -np.inf], np.nan)
train_df = train_df.fillna(np.nan)
train_df = train_df.dropna()

# Prepare data for training
y_train_data = train_df["class"] 
x_train_data = train_df.drop(["class", "latitude", "longitude", "spring_product_name", "autumn_product_name", "summer_product_name"], axis=1)
#pc_columns = pca(x_train_data,99)
pc_columns = sorted(["slope","aspect","dem","spring_evi2","spring_mndwi","spring_moisture","spring_ndbg","spring_ndre","spring_ndvi","spring_osavi","spring_AOT","spring_B01","spring_B02","spring_B03","spring_B04","spring_B05","spring_B06","spring_B07","spring_B08","spring_B09","spring_B11","spring_B12","spring_B8A","summer_evi2","summer_mndwi","summer_moisture","summer_ndbg","summer_ndre","summer_ndvi","summer_osavi","summer_AOT","summer_B01","summer_B02","summer_B03","summer_B04","summer_B05","summer_B06","summer_B07","summer_B08","summer_B09","summer_B11","summer_B12","summer_B8A","autumn_evi2","autumn_mndwi","autumn_moisture","autumn_ndbg","autumn_ndre","autumn_ndvi","autumn_osavi","autumn_AOT","autumn_B01","autumn_B02","autumn_B03","autumn_B04","autumn_B05","autumn_B06","autumn_B07","autumn_B08","autumn_B09","autumn_B11","autumn_B12","autumn_B8A"]) 
print(pc_columns)
reduced_x_train_data = train_df[pc_columns]
X_train, X_test, y_train, y_test = train_test_split(reduced_x_train_data, y_train_data, test_size=0.15)

# Train model 
clf = RandomForestClassifier(n_jobs=3)
clf.fit(X_train, y_train)
y_true = clf.predict(X_test)

labels=["agricola","beaches", "bosque", "bosqueRibera","cities","dehesas","matorral","pastos","plantacion","rocks","water","wetland"]
visualize_confusion_matrix(y_true, y_test, labels)

print(X_test.iloc[0:2,:],"\n", clf.predict(X_test.iloc[0:2,:]), y_test.iloc[0:2])


joblib.dump(clf, 'model.joblib')