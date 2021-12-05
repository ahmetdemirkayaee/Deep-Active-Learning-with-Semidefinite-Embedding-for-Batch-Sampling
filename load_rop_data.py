# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:45:50 2021

@author: byzkl
"""

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import pickle 
import os.path as ospath 

# Load data
folder_path = ospath.dirname(ospath.abspath(__file__))

partition_file_100=ospath.join(folder_path,'Partitions.p')
partition_file_100 = pickle.load(open(partition_file_100, 'rb'), encoding='latin1')
img_folder_100=ospath.join(folder_path,'preprocessed/All/')

part_rsd_test = partition_file_100['RSDTestPlusPartition']
label_rsd = partition_file_100['RSDLabels']
img_names = partition_file_100['orderName']
# print(len(img_names))

part_rsd_test = partition_file_100['RSDTestPlusPartition']
part_rsd_test = np.array([item for sublist in part_rsd_test for item in sublist])
ind_rsd_test = part_rsd_test.astype(np.int)
# choose binary absolute labels. 1: plus, 2:prep, 3:normal
rsd_labels_plus = np.zeros((label_rsd.shape[0],))
rsd_labels_plus[np.where(label_rsd == 1)[0]] = 1
rsd_labels_prep = 1. * rsd_labels_plus
rsd_labels_prep[np.where(label_rsd == 2)[0]] = 1

abs_thr='pre-plus'
if abs_thr == 'plus':
    label_rsd = rsd_labels_plus
else:
    label_rsd = rsd_labels_prep
# load test images for kthFold
img_test_list = [img_folder_100 + img_names[int(order + 1)] + '.png' for order in ind_rsd_test]
img_test_ori = img_to_array(load_img(img_test_list[0])).astype(np.uint8)[np.newaxis, :, :, :]

for img_name_iter in img_test_list[1:]:
    img_iter = img_to_array(load_img(img_name_iter)).astype(np.uint8)[np.newaxis, :, :, :]
    img_test_ori = np.concatenate((img_test_ori, img_iter), axis=0)
# Load test labels for kthFold
abs_imgs = img_test_ori
abs_labels = label_rsd[ind_rsd_test]
img_names_list = [name.split("All/")[1] for name in img_test_list]

print(img_names_list)
print(abs_labels)