import pandas as pd
import numpy as np
from utils import pca
import joblib
from sklearn.ensemble import RandomForestClassifier


# Temporal code to load premade dataset for training 


training = True
pc_columns = ['autumn_B06', 'autumn_evi', 'spring_B01', 'spring_B05', 'spring_evi', 'summer_B08', 'summer_WVP', 'summer_evi']

train_df = pd.read_csv("dataset.csv")
train_df = train_df.replace([np.inf, -np.inf], np.nan)
train_df = train_df.fillna(np.nan)
train_df = train_df.dropna()

# Prepare data for training
y_train_data = train_df["class"] 
x_train_data = train_df.drop("class", axis=1)
pc_columns = pca(x_train_data,98)
print(pc_columns)
reduced_x_train_data = train_df[pc_columns]

# Train model 
clf = RandomForestClassifier()
clf.fit(reduced_x_train_data, y_train_data)
joblib.dump(clf, 'model.pkl', compress=1)