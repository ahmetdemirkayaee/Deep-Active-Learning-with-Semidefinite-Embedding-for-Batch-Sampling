# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:06:57 2021

@author: byzkl
"""

import matplotlib.pyplot as plt
import numpy as np
# folder = "results/2class/"
folder = "results/2class/auc/"
plt.rcParams.update({'font.size': 15})
markers = ['.','o','v','^','<','>','8','s','p','P','*','H','X','D','x','h','+']
# lc = np.zeros((16,))
# for exp in range(0,3):
#     least_confidence = np.load("results/" + str(exp)+"_least_confidence.npy")
# # print(least_confidence.shape)
#     lc = np.add(lc,least_confidence)
# # print(lc)
# plt.plot(lc/3, label = "Least Confidence")

m=0

fig, ax =plt.subplots(figsize=(7,7))
lcd = np.zeros((5,16))
for exp in range(0,3):
    for j in range(1,5):
        alpha=j*0.25
        least_confidence_distance = np.load(folder + str(exp)+"_least_confidence_distance_"+str(alpha)+".npy")
        # least_confidence_distance = np.load(folder + str(exp)+"_margin_sampling_distance_"+str(alpha)+".npy")
        lcd[j,:] = np.add(lcd[j,:],least_confidence_distance)
    
for j in range(0,4):
    plt.plot(lcd[j,1:]/3, label = r"Least Confidence&Similarity(SDE) - ($\alpha$="+str(j*0.25)+")", marker=markers[m])
    m+=1
plt.plot(lcd[4,1:]/3, label = r"Least Confidence (SDE)", marker=markers[m])
m+=1

# # folder = "results/2class_pca/"
# folder = "results/3class_pca/"
# lcd = np.zeros((5,16))
# for exp in range(0,3):
#     for j in range(0,5):
#         alpha=j*0.25
#         # least_confidence_distance = np.load(folder + str(exp)+"_least_confidence_distance_"+str(alpha)+".npy")
#         least_confidence_distance = np.load(folder + str(exp)+"_margin_sampling_distance_"+str(alpha)+".npy")
#         lcd[j,:] = np.add(lcd[j,:],least_confidence_distance)
    
# for j in range(0,4):
#     plt.plot(lcd[j,1:]/3, label = r"Least Confidence&Similarity(PCA) - ($\alpha$="+str(j*0.25)+")", marker=markers[m])
#     m+=1
# plt.plot(lcd[4,1:]/3, label = r"Least Confidence (PCA)", marker=markers[m])
# m+=1
# # lcd = np.zeros((5,16))
# # for exp in range(0,3):
# #     for j in range(0,5):
# #         alpha=j*0.25
# #         least_confidence_distance = np.load(folder + str(exp)+"_margin_sampling_distance_"+str(alpha)+".npy")
# #         lcd[j,:] = np.add(lcd[j,:],least_confidence_distance)
    
# # for j in range(0,5):
# #     plt.plot(lcd[j,1:]/3, label = r"Margin Sampling Distance - ($\alpha$="+str(j*0.25)+")", marker=markers[m])
# #     m+=1
    
ax.legend(loc='upper right', bbox_to_anchor=(2.1,1))#loc = "best")
# plt.tight_layout()
# plt.rc('font', size=8) 
plt.xticks(range(least_confidence_distance.shape[0]))
plt.yticks(range(95,101,2))
# plt.title("Accuracy through Active Learning Cycles (2 Classes-1&2)")
plt.title("3 Classes (0&1&2)",pad=15)
# plt.title("2 Classes (1&2)",pad=15)
plt.xlabel("Active Learning Cycle")
plt.ylabel("Accuracy")
plt.grid()
# plt.savefig("3_pca_sde_least_confidence.png", dpi=100, bbox_inches='tight')
plt.savefig("3_least_confidence.png", dpi=100, bbox_inches='tight')
# plt.savefig("2_pca_sde_least_confidence.png", dpi=100, bbox_inches='tight')
# plt.savefig("3_pca_sde_margin_sampling.png", dpi=100, bbox_inches='tight')
plt.show()