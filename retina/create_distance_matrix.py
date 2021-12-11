# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:41:48 2021

@author: byzkl
"""

import pickle
import os
from os import walk
import os.path as ospath 
import pandas as pd
import numpy as np
import math

# Load data
folder_path = os.path.dirname(os.path.abspath(__file__))
# data = pd.read_pickle(ospath.join(folder_path, "mnist_scs_L120_K6_D2_data_800_by_784_03.28.41.015702.pkl")) # 2 classes
data = pd.read_pickle(ospath.join(folder_path, "retina_scs_L60_K6_D2_data_1000_by_50176_22.32.16.564495.pkl")) # 3 classes
print(data['Color'])
print(data['MVU Dimension 1'])

# Create distance matrix that shows the distance between each combination of two images
data_size = len(data)
distances = np.zeros((data_size,data_size))
for im1 in range(data_size):
    for im2 in range(data_size):
        im1_x = data['MVU Dimension 1'][im1]
        im1_y = data['MVU Dimension 2'][im1]
        im2_x = data['MVU Dimension 1'][im2]
        im2_y = data['MVU Dimension 2'][im2]
        distances[im1,im2] = math.sqrt((im1_x-im2_x)**2+(im1_y-im2_y)**2)
        
# with open(ospath.join(folder_path, "distances.npy"), 'wb') as f:
np.save(ospath.join(folder_path, "distances_retina.npy"), distances)