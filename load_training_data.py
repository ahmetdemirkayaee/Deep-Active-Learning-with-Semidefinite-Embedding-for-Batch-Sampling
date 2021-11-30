# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:35:56 2021

@author: byzkl
"""

import numpy as np
import os
import os.path as ospath

folder_path = os.path.dirname(os.path.abspath(__file__))
train_x = np.load(ospath.join(folder_path, "mnist_train_x.npy"))
print(train_x)