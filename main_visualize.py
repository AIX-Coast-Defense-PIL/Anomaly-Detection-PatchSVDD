import argparse
from pyexpat import model
import matplotlib.pyplot as plt
from codes import mvtecad
from tqdm import tqdm
from codes.utils import resize, makedirpath

import os

# gpu 지정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='ocean')
parser.add_argument('--ckpt', default='enchier.pkl')
parser.add_argument('--annotation', default=False)
args = parser.parse_args()


def save_maps(obj, maps):
    from skimage.segmentation import mark_boundaries
    N = maps.shape[0]
    images = mvtecad.get_x(obj, mode='test')

    if args.annotation:
        masks = mvtecad.get_mask(obj)

    for n in tqdm(range(N)):
        if args.annotation:
            fig, axes = plt.subplots(ncols=3)
            fig.set_size_inches(9, 3)

            image = resize(images[n], (128, 128))
            mask = resize(masks[n], (128, 128))
            image_marked = mark_boundaries(image, mask, color=(1, 0, 0), mode='thick')

            axes[0].imshow(image)
            axes[0].set_axis_off()

            axes[1].imshow(image_marked)
            axes[1].set_axis_off()

            axes[2].imshow(maps[n], vmax=maps[n].max(), cmap='Reds')
            axes[2].set_axis_off()
        else:
            fig, axes = plt.subplots(ncols=2)
            fig.set_size_inches(6, 3)

            image = resize(images[n], (128, 128))

            axes[0].imshow(image)
            axes[0].set_axis_off()

            print(maps[n].max())
            # axes[1].imshow(maps[n], vmax=maps[n].max(), cmap='Reds')
            axes[1].imshow(maps[n], vmax=5, cmap='Reds')
            axes[1].set_axis_off()

        plt.tight_layout()
        fpath = f'anomaly_maps/{obj}/{args.ckpt}/n{n:03d}.png'
        makedirpath(fpath)
        plt.savefig(fpath)
        plt.close()


#########################


def main():
    from codes.inspection import make_maps_NN_multiK
    from codes.networks import EncoderHier

    obj = args.obj
    ckpt = args.ckpt
    enc = EncoderHier(K=64, D=64).cuda()
    enc.load(obj, ckpt)
    enc.eval()
    # results = eval_encoder_NN_multiK(enc, obj)
    results = make_maps_NN_multiK(enc, obj)
    maps = results['maps_mult']

    save_maps(obj, maps)


if __name__ == '__main__':
    main()
  