# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:06:31 2019

@author: Helen
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 00:03:58 2019

@author: Helen
"""

#matplotlib inline
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
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        

GT_path = glob.glob("E:\\Helen\\Python_practice\\scarping_data2_train\\GroundTruth\\*.bmp") + glob.glob("E:\\Helen\\Python_practice\\scarping_data2_train\\GroundTruth\\*.png")
outpath1 = ("E:\\Helen\\Python_practice\\scarping_data2_train\\GT_hflip")
outpath1_2 = ("E:\\Helen\\Python_practice\\scarping_data2_train\\GT_hflip_bw")
filenames1 = os.listdir("E:\\Helen\\Python_practice\\scarping_data2_train\\groundTruth")

train_path = glob.glob("E:\\Helen\\Python_practice\\scarping_data2_train\\train_pre\\*.bmp") + glob.glob("E:\\Helen\\Python_practice\\scarping_data2_train\\train_pre\\*.png")
filenames2 = os.listdir("E:\\Helen\\Python_practice\\scarping_data2_train\\train_pre")
outpath2 = ("E:\\Helen\\Python_practice\\scarping_data2_train\\img_hflip")

                
for i, (image,mask) in enumerate(zip(train_path, GT_path)):
    image, mask=list((image,mask))
    filename2 = filenames2[i]
    filename2 = filename2[:-4]
#    filename = outpath + '\\' + filename + '.png'#os.path.join(outpath, filename + '.png')
    filename2 = os.path.join(outpath2, 'Db5_1_'+ str(i) + '.jpg')
    n = cv2.imread(image)
    image = np.asarray(n)

    filename1 = filenames1[i]
    filename1 = filename1[:-4]
    #    filename = outpath + '\\' + filename + '.png'#os.path.join(outpath, filename + '.png')
    filename1 = os.path.join(outpath1, 'Db5_1_'+ str(i) + '.png')
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
          



    
              


