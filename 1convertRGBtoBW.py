# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:05:56 2019

@author: Helen
"""
    
#save img into binary image folder
import numpy as np
import cv2
import glob
import os


images=[]

import cv2 as cv

# GT_path: to read all image have extension are .bmp and .png
#outpath1: address for DT_db
#outpath1_2: adress for groundtruth after convert from RGBtoBW
#filename1: list of filename of image in groundTruth_pre folder

GT_path = glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\groundTruth_pre\\*.bmp") + glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\groundTruth_pre\\*.png")
outpath1_PNG = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\GT_db")
outpath1_2 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\GT_bw")
filenames1 = os.listdir("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\groundTruth_pre")

train_path = glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\train_pre\\*.bmp") + glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\train_pre\\*.png")
filenames2 = os.listdir("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\train_pre")
outpath2 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\train_db")
outpath2_png = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset_augmented\\Scraping_Data2\\train_dbPNG")



for i, img in enumerate(GT_path):
    filename1 = filenames1[i]
    filename1 = filename1[:-4]
    filename1_2 = filenames1[i]
    filename1_2 = filename1_2[:-4]
    filename1_jpg = filenames1[i]
    filename1_jpg = filename1_jpg[:-4]
#    filename = outpath + '\\' + filename + '.png'#os.path.join(outpath, filename + '.png')
    filename1 = os.path.join(outpath1_PNG, filename1 + '.png')


    print(filename1)
    n = cv.imread(img,2)
    img = np.asarray(n)
    ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    bw_img2=bw_img/255
    
    filename1_2 = os.path.join(outpath1_2, filename1_2 + '.png')
    print('i: {}'.format(i))
    cv2.imwrite(filename1, bw_img)
    cv2.imwrite(filename1_2, bw_img2)

    
for i, img in enumerate(train_path):
  # creat filename for image.jpg
    filename2 = filenames2[i]
    filename2 = filename2[:-4]
    #creat filename for image.png
    filename2_png = filenames2[i] 
    filename2_png = filename2_png[:-4]
#    filename = outpath + '\\' + filename + '.png'#os.path.join(outpath, filename + '.png')
    filename2 = os.path.join(outpath2, filename2 + '.jpg')
    filename2_png = os.path.join(outpath2_png, filename2_png + '.png')
    print(filename2)
    n = cv.imread(img)
    img = np.asarray(n)

    print('i: {}'.format(i))
    cv2.imwrite(filename2, img)
    cv2.imwrite(filename2_png, img)
    
test_path = glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\test_pre\\*.bmp")
filenames3 = os.listdir("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\test_pre")
outpath3 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\test_db")

for i, img in enumerate(test_path):
    filename3 = filenames3[i]
    filename3 = filename3[:-4]
#    filename = outpath + '\\' + filename + '.png'#os.path.join(outpath, filename + '.png')
    filename3 = os.path.join(outpath3, filename3 + '.jpg')
    print(filename3)
    n = cv.imread(img)
    img = np.asarray(n)

    print('i: {}'.format(i))
    cv2.imwrite(filename3, img)
    
    
## for testing of each database
#GTtest_db1 = glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\Db1_FgBlue_BgMetal\Test\Ground_Truth\\*.bmp")
#filenames1 = os.listdir("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\Db1_FgBlue_BgMetal\Test\Ground_Truth")
#outpath1 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\Db1_FgBlue_BgMetal\\Test\\GT_fortestdb1")
#
#test_db1 = glob.glob("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\Db1_FgBlue_BgMetal\Test\Scraping\\*.bmp")
#filenames2 = os.listdir("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\Db1_FgBlue_BgMetal\Test\Scraping")
#outpath2 = ("E:\\A_paper_thesis\\paper5\\tensorflow_deeplabv3plus_scrapingData\\dataset\\Scraping_Data2\\Db1_FgBlue_BgMetal\\Test\\train_fortestdb1")
#
#
#for i, img in enumerate(GTtest_db1):
#    filename1 = filenames1[i]
#    filename1 = filename1[:-4]
#    filename1_2 = filenames1[i]
#    filename1_2 = filename1_2[:-4]
##    filename = outpath + '\\' + filename + '.png'#os.path.join(outpath, filename + '.png')
#    filename1 = os.path.join(outpath1, filename1 + '.png')
#
#    print(filename1)
#    n = cv.imread(img,2)
#    img = np.asarray(n)
#    ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#    bw_img2=bw_img/255
#    
#    print('i: {}'.format(i))
#    cv2.imwrite(filename1, bw_img2)
#    
#for i, img in enumerate(test_db1):
#    filename2 = filenames2[i]
#    filename2 = filename2[:-4]
##    filename = outpath + '\\' + filename + '.png'#os.path.join(outpath, filename + '.png')
#    filename2 = os.path.join(outpath2, filename2 + '.jpg')
#    print(filename2)
#    n = cv.imread(img)
#    img = np.asarray(n)
#
#    print('i: {}'.format(i))
#    cv2.imwrite(filename2, img)

    