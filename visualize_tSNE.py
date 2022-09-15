import argparse
from pyexpat import model
import matplotlib.pyplot as plt
from codes import mvtecad
from tqdm import tqdm
from codes.utils import resize, makedirpath
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

import os

# gpu 지정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='ocean_partial01')
parser.add_argument('--ckpt', default='basic_ep295_ac0.pkl')
parser.add_argument('--annotation', default=False)
args = parser.parse_args()

SAVE_DIR = '/home/hajung/Anomaly-Detection-PatchSVDD-PyTorch/embedding_features'

def eval_and_save_features(obj, ckpt, save_path=None):
    from codes.inspection import eval_encoder
    from codes.networks import EncoderHier

    enc = EncoderHier(K=64, D=64).cuda()
    enc.load(obj, ckpt)
    enc.eval()
    results = eval_encoder(enc, obj)

    if save_path:
        np.save(save_path, results)

    return results


def visualize_tSNE(data, save_path):
    data_num = data['embs64_te'].shape[0]
    data_01 = pd.DataFrame(data['embs64_te'].reshape(data_num, -1))
    data_01['label'] = '64-test'
    print('-------------\n64-test')
    print(data_01)

    data_num = data['embs64_tr'].shape[0]
    data_02 = pd.DataFrame(data['embs64_tr'].reshape(data_num, -1))
    data_02['label'] = '64-train'
    print('--------------\n64-train')
    print(data_02)

    data_all = pd.concat([data_01, data_02], ignore_index=True)
    print('-------------\nall')
    print(data_all)

    n_components = 2                             # 축소 시 차원
    model = TSNE(n_components=n_components)
    transformed = model.fit_transform(data_all[[i for i in range(10816)]])  # label 제외하고 입력

    fig, ax = plt.subplots()

    xs = transformed[:,0]
    ys = transformed[:,1]
    scatter = ax.scatter(xs,ys, c=data_all['label'])

    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)

    plt.savefig(save_path+'.png')
    plt.close()

def main():
    obj = args.obj
    ckpt = args.ckpt

    save_path = os.path.join(SAVE_DIR, obj, ckpt)
    makedirpath(save_path)

    saved = True
    if not saved:
        result = eval_and_save_features(obj, ckpt, save_path)      # make new emgedding features
    else:
        results = np.load(save_path+'.npy', allow_pickle=True)     # load saved embedding feautres
        results = results.all()

    visualize_tSNE(results, save_path)



if __name__ == '__main__':
    main()
  