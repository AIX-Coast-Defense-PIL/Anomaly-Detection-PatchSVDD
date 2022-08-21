import os
import shutil
from glob import glob
import pandas as pd
import random

DATASET_PATH = '/home/hajung/Anomaly-Detection-PatchSVDD-PyTorch/datasets/ocean/'

def move_by_exel(DATASET_PATH):
    labels = pd.read_excel(os.path.join(DATASET_PATH, 'class_labels.xlsx'))

    for i in range(492):
        if  labels.loc[i]['부표'] == 1 :
            filename = str(int(labels.loc[i][0])).zfill(4)+'.jpg'
            shutil.move(DATASET_PATH + 'train/' + filename, DATASET_PATH + 'test/buoy/' + filename)
        elif labels.loc[i]['고무보트'] == 1:
            filename = str(int(labels.loc[i][0])).zfill(4)+'.jpg'
            shutil.move(DATASET_PATH + 'train/' + filename, DATASET_PATH + 'test/rubber_boat/' + filename)
        elif labels.loc[i]['미상'] == 1:
            filename = str(int(labels.loc[i][0])).zfill(4)+'.jpg'
            shutil.move(DATASET_PATH + 'train/' + filename, DATASET_PATH + 'test/unknown/' + filename)

def move_goods_randomly(DATASET_PATH):
    fpattern = os.path.join(DATASET_PATH, f'train/good/*')
    fpaths = sorted(glob(fpattern))
    fpaths = random.sample(fpaths, 20)
    for fpath in fpaths:
        next_path = os.path.join(DATASET_PATH, 'test/good/' + fpath.split('/')[-1])
        shutil.move(fpath, next_path)


if __name__ =='__main__':
    # move_by_exel(DATASET_PATH)
    move_goods_randomly(DATASET_PATH)