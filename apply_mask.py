## https://stackoverflow.com/questions/66095686/apply-a-segmentation-mask-through-opencv
from skimage.segmentation import mark_boundaries
from codes import mvtecad
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='ocean_all')
args = parser.parse_args()

SAVE_PATH = "/home/hajung/Anomaly-Detection-PatchSVDD-PyTorch/datasets/ocean_all/train/masked"

# remove ocean & sky
def apply_mask(obj):
    
    images = mvtecad.get_x(obj, mode='train')
    masks = mvtecad.get_MaSTr1325_mask(obj)

    masks = np.array([[[[r, r, r] for r in row] for row in mask] for mask in masks])     # rgb에 맞게 차원 확대

    for i in range(len(images)):
        image_masked = np.where(masks[i], 0, images[i])   # remove background
        image_masked = Image.fromarray(image_masked)
        image_masked.save(os.path.join(SAVE_PATH, str(i).zfill(4)+'.png'))




def main():
    apply_mask(args.obj)


if __name__ =='__main__':
    main()