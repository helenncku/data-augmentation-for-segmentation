# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:03:27 2019

@author: Helen
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)
 
# train_image for data augmentation       
GT_path = glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\GT_forAug\\*.png")
filenames1 = os.listdir("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\GT_forAug")

outpath1 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\data_other\\GT_augmented\\GT_1hflip")
outpath1_2 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\data_other\\GT_augmented\\GT_2rotate90")
outpath1_3 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\data_other\\GT_augmented\\GT_3transpose")
outpath1_4 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\data_other\\GT_augmented\\GT_4compose")

train_path = glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\train_forAug\\*.jpg")
filenames2 = os.listdir("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\train_forAug")

outpath2 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\dataset_augmented\\Scraping_Data2\\data_other\\img_augmented\\img_1hflip")
outpath2_2 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\dataset_augmented\\Scraping_Data2\\data_other\\img_augmented\\img_2rotate90")
outpath2_3 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\dataset_augmented\\Scraping_Data2\\data_other\\img_augmented\\img_3transpose")
outpath2_4 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\dataset_augmented\\Scraping_Data2\\data_other\\img_augmented\\img_4compose")

   
for i, (image,mask) in enumerate(zip(train_path, GT_path)):
    image, mask=list((image,mask))
    
    # processing for train image
    filename2 = filenames2[i]
    filename2 = filename2[:-4]

    filename2 = os.path.join(outpath2, 'hflip_'+ str(i) + '.jpg')
    filename2_2 = os.path.join(outpath2_2, 'rotate_'+ str(i) + '.jpg')
    filename2_3 = os.path.join(outpath2_3, 'transpose_'+ str(i) + '.jpg')
    filename2_4 = os.path.join(outpath2_4, 'compose_'+ str(i) + '.jpg')

    n = cv2.imread(image)
    image = np.asarray(n)
    original_height, original_width = image.shape[:2]
    
    # data processing for ground truth
    filename1 = filenames1[i]
    filename1 = filename1[:-4]
    
    filename1 = os.path.join(outpath1, 'hflip_'+ str(i) + '.png')
    filename1_2 = os.path.join(outpath1_2, 'rotate_'+ str(i) + '.png')
    filename1_3 = os.path.join(outpath1_3, 'transpose_'+ str(i) + '.png')
    filename1_4 = os.path.join(outpath1_4, 'compose_'+ str(i) + '.png')  
    
    n1 = cv2.imread(mask,2)
    mask = np.asarray(n1)
    
# data augmentation    
    aug = HorizontalFlip(p=1)
    augmented = aug(image=image, mask=mask) #problem here
    
    image_h_flipped=augmented['image']
    mask_h_flipped=augmented['mask']
            
    #save image_flipped and their mask
    cv2.imwrite(filename2,image_h_flipped)        
    cv2.imwrite(filename1,mask_h_flipped)      
            
    # 2. random rotate 90
    aug = RandomRotate90(p=1)
    
    augmented = aug(image=image, mask=mask)
            
    image_rot90 = augmented['image']
    mask_rot90 = augmented['mask']
            
    cv2.imwrite(filename2_2,image_rot90)  
    cv2.imwrite(filename1_2,mask_rot90)
            
#     3. Transpose       
    aug = Transpose(p=1)
    
    augmented = aug(image=image, mask=mask)
            
    image_transposed = augmented['image']
    mask_transposed = augmented['mask']
            
    cv2.imwrite(filename2_3,image_transposed)  
    cv2.imwrite(filename1_3,mask_transposed)
            
    # 4. Compose
    
    aug = Compose([
        OneOf([RandomSizedCrop(min_max_height=(50, 90), height=original_height, width=original_width, p=0.5),
              PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),    
        VerticalFlip(p=0.5),              
        RandomRotate90(p=0.5),
        OneOf([ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
              GridDistortion(p=0.5),
              OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
               ], p=0.8),
        CLAHE(p=0.8),
        RandomBrightnessContrast(p=0.8),    
        RandomGamma(p=0.8)])
        
    augmented = aug(image=image, mask=mask)
            
    image_heavy = augmented['image']
    mask_heavy = augmented['mask']
            
    cv2.imwrite(filename2_4,image_heavy)  
    cv2.imwrite(filename1_4,mask_heavy)
    
    # 5. 
