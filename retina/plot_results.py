# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:06:57 2021

@author: byzkl
"""

import matplotlib.pyplot as plt
import numpy as np
folder = "results/2class/"
# folder = "results/3class/"
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
    for j in range(0,5):
        alpha=j*0.25
        least_confidence_distance = np.load(folder + str(exp)+"_least_confidence_distance_"+str(alpha)+".npy")
        lcd[j,:] = np.add(lcd[j,:],least_confidence_distance)
max_least = np.where(lcd[:,-1]==np.amax(lcd[:,-1]))

    
# for j in range(0,4):
#     plt.plot(lcd[j,:]/3, label = r"Least Confidence&Similarity - ($\alpha$="+str(j*0.25)+")", marker=markers[m])
#     m+=1
# plt.plot(lcd[4,:]/3, label = r"Least Confidence", marker=markers[m])
# m+=1

max_least = max_least[0][0]
print(max_least)
if max_least !=4:
    plt.plot(lcd[max_least,:]/3, label = r"Least Confidence&Similarity - ($\alpha$="+str(max_least*0.25)+")", marker=markers[m])
    m+=1
else:
    plt.plot(lcd[max_least,:]/3, label = r"Least Confidence", marker=markers[m])
m+=1

lcd = np.zeros((5,16))
for exp in range(0,3):
    for j in range(0,5):
        alpha=j*0.25
        least_confidence_distance = np.load(folder + str(exp)+"_entropy_distance_"+str(alpha)+".npy")
        lcd[j,:] = np.add(lcd[j,:],least_confidence_distance)
max_entropy = np.where(lcd[:,-1]==np.amax(lcd[:,-1]))
    
# for j in range(0,4):
#     plt.plot(lcd[j,:]/3, label = r"Entropy&Similarity - ($\alpha$="+str(j*0.25)+")", marker=markers[m])
#     m+=1
# plt.plot(lcd[4,:]/3, label = r"Entropy", marker=markers[m])
# m+=1

max_entropy =max_entropy[0][0]
if max_entropy !=4:
    plt.plot(lcd[max_entropy,:]/3, label = r"Entropy&Similarity - ($\alpha$="+str(max_entropy*0.25)+")", marker=markers[m])
    m+=1
else:
    plt.plot(lcd[max_entropy,:]/3, label = r"Entropy", marker=markers[m])
m+=1

lcd = np.zeros((5,16))
for exp in range(0,3):
    for j in range(0,5):
        alpha=j*0.25
        least_confidence_distance = np.load(folder + str(exp)+"_margin_sampling_distance_"+str(alpha)+".npy")
        lcd[j,:] = np.add(lcd[j,:],least_confidence_distance)
max_margin = np.where(lcd[:,-1]==np.amax(lcd[:,-1]))
    
# for j in range(0,4):
#     plt.plot(lcd[j,:]/3, label = r"Margin Sampling&Similarity - ($\alpha$="+str(j*0.25)+")", marker=markers[m])
#     m+=1
# plt.plot(lcd[4,:]/3, label = r"Margin Sampling", marker=markers[m])
# m+=1

max_margin = max_margin[0][0]
if max_margin !=4:
    plt.plot(lcd[max_margin,:]/3, label = r"Margin Sampling&Similarity - ($\alpha$="+str(max_margin*0.25)+")", marker=markers[m])
    m+=1
else:
    plt.plot(lcd[max_margin,:]/3, label = r"Margin Sampling", marker=markers[m])
m+=1
    

lcd = np.zeros((1,16))
j = 0
for exp in range(0,3):
    alpha=j*0.25
    least_confidence_distance = np.load(folder + str(exp)+"_random_"+str(alpha)+".npy")
    lcd[j,:] = np.add(lcd[j,:],least_confidence_distance)
# max_margin = np.where(lcd[:,-1]==np.amax(lcd[:,-1]))
j=0
plt.plot(lcd[j,:]/3, label = r"Random Sampling", marker=markers[m])
m+=1

ax.legend(loc='upper right')#loc='upper right', bbox_to_anchor=(2,1))#loc = "best")
# ax.legend(loc='upper right', bbox_to_anchor=(2,1))#loc = "best")
# plt.tight_layout()
# plt.rc('font', size=8) 
plt.xticks(range(least_confidence_distance.shape[0]))
plt.yticks(range(60,101,2))
# plt.title("Accuracy through Active Learning Cycles (2 Classes-1&2)")
# plt.title("3 Classes (0&1&2)",pad=15)
plt.title("2 Classes (Sick&Healthy)",pad=15)
plt.xlabel("Active Learning Cycle")
plt.ylabel("Accuracy")
plt.grid()
# plt.savefig("2_least_confidence_entropy.png", dpi=100, bbox_inches='tight')
# plt.savefig("results/2class_plots/2_least_confidence.png", dpi=100, bbox_inches='tight')
# plt.savefig("results/2class_plots/2_margin_sampling.png", dpi=100, bbox_inches='tight')
plt.savefig("results/2class_plots/2_least_confidence_entropy_margin_sampling.png", dpi=100, bbox_inches='tight')
# plt.savefig("results/2class_plots/2_entropy.png", dpi=100, bbox_inches='tight')
# plt.savefig("results/3class_plots/zoomed_3_margin_sampling.png", dpi=100, bbox_inches='tight')
# plt.savefig("results/3class_plots/zoomed_3_least_confidence_entropy_margin_sampling.png", dpi=100, bbox_inches='tight')
plt.show()