Prrprocessing data

1: - run convertRGBtoBW.py to convert Groundtruth_pre from RGB image to binary imaage save to (GT_db)
   - then save groundtruth image to .png format and train_pre image to .jpg save to train_db (same as 
   deeplabv3+ (https://github.com/rishizek/tensorflow-deeplab-v3-plus))
2: - run creat_txt_file.py to get filename of all image in GT_db and train_db
   - and save as groundtruth_pre.txt and train_pre.txt
3: - run creat_train_val_shuffle.py to randomly shuffle 
   - data (train_pre.txt and groundtruth_pre.txt),
   - split train_pre.txt into train_main.txt and val_main.txt