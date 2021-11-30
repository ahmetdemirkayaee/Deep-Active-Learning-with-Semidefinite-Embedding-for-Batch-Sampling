# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:06:57 2021

@author: byzkl
"""

import matplotlib.pyplot as plt
import numpy as np

lc = np.zeros((30,))
for exp in range(1,4):
    least_confidence = np.load("results/" + str(exp)+"_least_confidence.npy")
# print(least_confidence.shape)
    lc = np.add(lc,least_confidence)
print(lc)
plt.plot(lc/3, label = "Least Confidence")

lcd = np.zeros((30,))
for exp in range(1,4):
    least_confidence_distance = np.load("results/" + str(exp)+"_least_confidence_distance.npy")
    lcd = np.add(lcd,least_confidence_distance)
plt.plot(lcd/3, label = "Least Confidence Distance")

plt.legend(bbox_to_anchor=(0.5, -0.2), loc = "best")
# plt.tight_layout()
plt.xticks(range(least_confidence.shape[0]))
plt.show()