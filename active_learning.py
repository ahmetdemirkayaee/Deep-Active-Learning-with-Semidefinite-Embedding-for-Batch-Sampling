# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:21:21 2021

@author: byzkl
"""
import math
import numpy as np
import numpy.ma as ma

def calc_entropy(predictions,num_of_classes):
    ent = 0
    for i in range(num_of_classes):
        ent -= predictions[i] * math.log(predictions[i],10)
    return ent

# def calc(predictions,num_of_classes):
#     ent = 0
#     for i in range(num_of_classes):
#         ent -= predictions[i] * math.log(predictions[i],10)
#     return ent


def choose_sample(samples,num_of_classes,acquisition,num_of_selection):
    # print("trial", samples.shape)
    if acquisition=='entropy':
        entropies = {}
        for i, sample in enumerate(samples):
            ent = calc_entropy(sample,num_of_classes)
            entropies[i] = ent
        print(entropies)
        entropies = sorted(entropies.items(), key=lambda item: item[1], reverse=True)
        entropies = [i[0] for i in entropies]
        entropies = entropies[:num_of_selection]
        
        return entropies
    elif acquisition=='least confident':
        confs = {}
        for i, sample in enumerate(samples):
            conf = np.max(sample)
            confs[i] = conf
        confs = sorted(confs.items(), key=lambda item: item[1])
        confs = [i[0] for i in confs]
        confs = confs[:num_of_selection]
        print(confs)
        return confs
    elif acquisition=='margin sampling':
        margins = {}
        for i, sample in enumerate(samples):
            predictions = np.copy(sample)
            conf1 = np.max(predictions)
            mask = predictions == conf1
            predictions = ma.masked_array(predictions, mask = mask)
            # predictions.remove(conf1)
            conf2 = np.max(predictions)
            print(conf1,conf2)
            margins[i] = conf1-conf2
        margins = sorted(margins.items(), key=lambda item: item[1])
        margins = [i[0] for i in margins]
        margins = margins[:num_of_selection]
        print(margins)
        return margins

def choose_sample_distances(samples,num_of_classes,acquisition,num_of_selection,distances):
    if acquisition=='entropy':
        entropies = {}
        for i, sample in enumerate(samples):
            ent = calc_entropy(sample,num_of_classes)
            entropies[i] = ent
        entropies = sorted(entropies.items(), key=lambda item: item[1], reverse=True)
        entropies = [i[0] for i in entropies]
        entropies = entropies[:num_of_selection]
        print(entropies)
        return entropies
    elif acquisition=='least confident':
        confs = {}
        for i, sample in enumerate(samples):
            conf = np.max(sample)
            confs[i] = conf       
        confs = sorted(confs.items(), key=lambda item: item[1])
        # batch created
        batch = []
        # first element is appended
        batch.append(confs[0][0])
        # print(confs[0])
        max_so_far = -1
        max_so_far_ = []
        for n in range(1,num_of_selection):
            # print("For " + str(n+1) + "th selection")
            for i in confs:
                dist = 0
                for s in batch:
                    dist += distances[s,i[0]]
                # print(dist)
                acq = i[1]+dist
                info = [i[0], i[1]]
                info.append(dist)
                if acq>max_so_far and i[0] not in batch:
                    max_so_far = acq
                    max_so_far_ = info
            # print(max_so_far_)
            batch.append(max_so_far_[0])
        # confs = [i[0] for i in confs]
        # confs = confs[:num_of_selection]
        print(batch)
        return batch
    elif acquisition=='margin sampling':
        margins = {}
        for i, sample in enumerate(samples):
            predictions = np.copy(sample)
            conf1 = np.max(predictions)
            mask = predictions == conf1
            predictions = ma.masked_array(predictions, mask = mask)
            # predictions.remove(conf1)
            conf2 = np.max(predictions)
            print(conf1,conf2)
            margins[i] = conf1-conf2
        margins = sorted(margins.items(), key=lambda item: item[1])
        margins = [i[0] for i in margins]
        margins = margins[:num_of_selection]
        print(margins)
        return margins    
    
# samples  = [[0.4,0.6],[0.5,0.5],[0.3,0.7]]    
# # predictions = [0.4,0.6]
# num_of_classes = 2
# # print(calc_entropy(predictions,num_of_classes))
# # choose_sample(samples,num_of_classes,'entropy')
# # choose_sample(samples,num_of_classes,'least confident')
# choose_sample(samples,num_of_classes,'margin sampling')