# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:51:42 2019

@author: Helen
"""


import os

#WD = "E:\\A_paper_thesis\\paper5\\Ford_dataset\\Scraping Data2\\Train_db"
#files = glob.glob(os.path.join(WD, '*.bmp'))
#with open('train.txt', 'w') as in_files:
#    in_files.writelines(fn + '\n' for fn in files)
    
 
#files_no_ext = [".".join(f.split(".")[:-1]) for f in os.listdir() if os.path.isfile(f)]
#print(files_no_ext)
#list_ = ['.'.join(x.split('.')[:-1]) for x in os.listdir("path/to/Data") if os.path.isfile(os.path.join('path/to/Data', x))]

WD = "E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\train_db"
#WD1="E:\\A_paper_thesis\\paper5\\Ford_dataset\\Scraping Data2\\GroundTruth_db"
files = [".".join(f.split(".")) for f in os.listdir(WD) if os.path.isfile(os.path.join(WD, f))]
with open('train_pre.txt', 'w') as in_files:
    in_files.writelines(fn + '\n' for fn in files)


WD1="E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\GT_bw"
files2 = [".".join(f.split(".")) for f in os.listdir(WD1) if os.path.isfile(os.path.join(WD1, f))]
with open('groundtruth_pre.txt', 'w') as in_files:
    in_files.writelines(fn + '\n' for fn in files2)
    
WD2="E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\test_db"
files3 = [".".join(f.split(".")) for f in os.listdir(WD2) if os.path.isfile(os.path.join(WD2, f))]
with open('test_sample.txt', 'w') as in_files:
    in_files.writelines(fn + '\n' for fn in files3)
    
    
