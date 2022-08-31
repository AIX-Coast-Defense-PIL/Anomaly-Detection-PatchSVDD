import os
import shutil
from glob import glob
import pandas as pd
import random

DATASET_PATH = '/home/hajung/Anomaly-Detection-PatchSVDD-PyTorch/datasets/ocean_all/'

# exel 파일을 활용해, 부표, 고무보트, 미상 물체가 있는 이미지를 다른 이미지와 분리
def divide_by_exel(DATASET_PATH):
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

def divide_by_exel_index(DATASET_PATH):
    labels = pd.read_excel(os.path.join(DATASET_PATH, 'masked_class_labels.xlsx'))

    for i in range(len(labels['object'])):
        if  labels.loc[i]['object'] == 0 :
            filename = str(int(labels.loc[i][0])).zfill(4)+'.png'
            shutil.move(DATASET_PATH + 'train/masked/' + filename, DATASET_PATH + 'train/background/' + filename)
        elif labels.loc[i]['object'] == 1:
            filename = str(int(labels.loc[i][0])).zfill(4)+'.png'
            shutil.move(DATASET_PATH + 'train/masked/' + filename, DATASET_PATH + 'test/rubber_boat/' + filename)
        elif  labels.loc[i]['object'] == 2 :
            filename = str(int(labels.loc[i][0])).zfill(4)+'.png'
            shutil.move(DATASET_PATH + 'train/masked/' + filename, DATASET_PATH + 'test/buoy/' + filename)
        else:
            filename = str(int(labels.loc[i][0])).zfill(4)+'.png'
            shutil.move(DATASET_PATH + 'train/masked/' + filename, DATASET_PATH + 'test/unknown/' + filename)

# good 이미지 중 20개를 랜덤하게 이동
def move_goods_randomly(DATASET_PATH):
    fpattern = os.path.join(DATASET_PATH, f'train/background/*')
    fpaths = sorted(glob(fpattern))
    fpaths = random.sample(fpaths, 20)
    for fpath in fpaths:
        next_path = os.path.join(DATASET_PATH, 'test/background/' + fpath.split('/')[-1])
        shutil.move(fpath, next_path)


# 전체 이미지 중 원하는 이미지만 이동
# '엑셀 파일 특정 column에 있는 숫자 = 파일 명' 인 이미지 지동
def pick_images_by_exel(DATASET_PATH):
    labels = pd.read_excel(os.path.join(DATASET_PATH, 'background_labels.xlsx'))
    img_names = list(labels['Unnamed: 2'])
    img_names = [img_name for img_name in img_names if pd.isnull(img_name) == False]
    file_names = []
    for img_name in img_names:
        if img_name[0] == 'n' : #and int(img_name[1:]) > 662:
            file_name = img_name[1:]+'.jpg'
            file_names.append(file_name)
        # else:
        #     file_name = img_name+'.jpg'
        
    # print(file_names)

    for file_name in file_names:
        origin_path = os.path.join(DATASET_PATH, 'all', file_name)
        next_path  = os.path.join(DATASET_PATH, 'train/left', file_name)
        shutil.move(origin_path, next_path)

if __name__ =='__main__':
    # divide_by_exel(DATASET_PATH)
    # move_goods_randomly(DATASET_PATH)
    # pick_images_by_exel('/home/hajung/Anomaly-Detection-PatchSVDD-PyTorch/datasets/ocean_partial01/')
    divide_by_exel_index('/home/hajung/Anomaly-Detection-PatchSVDD-PyTorch/datasets/ocean_all/')