# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:23:12 2019

@author: Helen
"""

import random

with open("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\train_pre.txt","r") as infile:
    file1 = infile.readlines()
with open("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\groundtruth_pre.txt","r") as infile:
    file2 = infile.readlines()

c = list(zip(file1,file2))

random.shuffle(c)
train_pre, groundtruth = zip(*c)



#with open('train_pre2.txt', 'w') as outfile:
#    outfile.writelines(train_pre)
#    
#with open('GT_pre2.txt', 'w') as outfile:
#    outfile.writelines(groundtruth)
    
# split val and train
    
from math import floor
split = 0.7
split_index = floor(len(train_pre) * split)
training = train_pre[:split_index]
val = train_pre[split_index:]
    
with open('train_main.txt', 'w') as outfile:
    outfile.writelines(training)
    
with open('val_main.txt', 'w') as outfile:
    outfile.writelines(val)




    
    




