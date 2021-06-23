# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:13:00 2021

@author: Carter
"""

from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np

np.random.seed(320)

encoder = OneHotEncoder(sparse = False)

train_x = preprocessing.normalize(pd.read_csv("../Data/train_x_seeds.csv"))
train_y = encoder.fit_transform(pd.read_csv("../Data/train_y_seeds.csv"))
test_x = preprocessing.normalize(pd.read_csv("../Data/test_x_seeds.csv"))
test_y = encoder.fit_transform(pd.read_csv("../Data/test_y_seeds.csv"))

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=19, activation='relu', kernel_initializer = "uniform"))
model.add(keras.layers.Dense(32, activation = 'sigmoid', kernel_initializer = "uniform"))
model.add(keras.layers.Dense(16, activation = 'softmax', kernel_initializer = "uniform"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(train_x, train_y, epochs = 150, batch_size = 10)

_, accuracy = model.evaluate(train_x, train_y)
_, test_acc = model.evaluate(test_x, test_y)

model_json = model.to_json()
with open("seed_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("seed_model.h5")

with open("seed_model.txt", "w+") as f:
    f.write(str(accuracy))
    f.write("\n")
    f.write("\n")
    f.write(str(test_acc))
    f.close()

validate = pd.read_csv("../Data/cbb20FINAL.csv")
post_2020 = pd.DataFrame(model.predict_classes(validate.iloc[:,1:]))
withTeams = pd.concat([validate.iloc[:,0], post_2020], axis = 1)
withTeams.to_csv("../Data/cbb20SEEDS.csv")

