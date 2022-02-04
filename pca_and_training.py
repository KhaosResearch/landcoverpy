import pandas as pd
import numpy as np
from utils import pca
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
pc_columns = pca(x_train_data,99)
print(pc_columns)
reduced_x_train_data = train_df[pc_columns]
X_train, X_test, y_train, y_test = train_test_split(reduced_x_train_data, y_train_data, test_size=0.15)

# Train model 
clf = RandomForestClassifier(n_jobs=32)
clf.fit(X_train, y_train)
y_true = clf.predict(X_test)
print(confusion_matrix(y_true, y_test))
print(X_test.iloc[0:2,:],"\n", clf.predict(X_test.iloc[0:2,:]), y_test.iloc[0:2])


joblib.dump(clf, 'model.joblib')