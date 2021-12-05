# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:20:18 2021

@author: byzklknl
"""

import pickle
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os.path as ospath 

# Load data
folder_path = ospath.dirname(ospath.abspath(__file__))
partition_file_6000 = ospath.join(folder_path,'6000Partitions.p')
partition_file_6000 = pickle.load(open(partition_file_6000, 'rb'), encoding='latin1')
# print(folder_path)
img_folder_6000=ospath.join(folder_path,'preprocessed_JamesCode/')

part_rsd_test = partition_file_6000['RSDTestPartition']
img_names = partition_file_6000['imgNames']
rsd_labels_plus = partition_file_6000['RSDLabelsPlus']
rsd_labels_prep = partition_file_6000['RSDLabelsPreP']
# test on all 5000
ind_rsd_test = part_rsd_test[0].astype(np.int)
for k in [1, 2, 3, 4]:
    ind_rsd_test = np.append(ind_rsd_test, part_rsd_test[k].astype(np.int))
abs_thr='pre-plus'
if abs_thr == 'plus':
    abs_labels = rsd_labels_plus[ind_rsd_test]
else:
    abs_labels = rsd_labels_prep[ind_rsd_test]
img_test_list = [img_folder_6000 + img_names[int(order)] + '.png' for order in ind_rsd_test]
abs_imgs = img_to_array(load_img(img_test_list[0]))[np.newaxis, :, :, :]
# print(len(img_test_list))
i=0
for img_name_iter in img_test_list[1:10]:
    print(i)
    img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
    abs_imgs = np.concatenate((abs_imgs, img_iter), axis=0)
    i+=1
    
img_names_list = [name.split("preprocessed_JamesCode/")[1] for name in img_test_list]

print(img_names_list)
print(abs_labels)