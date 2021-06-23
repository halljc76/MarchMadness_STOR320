# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:02:37 2021

@author: Carter
"""

from ann_visualizer.visualize import ann_viz;
from tensorflow.keras.models import model_from_json
import numpy
numpy.random.seed(320)

def getNN(title, file, h5file):
    json_file = open(file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(h5file)
    ann_viz(model, title=title)
    
getNN("Postseason Berth Neural Net Architecture", 
      "post_model.json", 
      "post_model.h5")