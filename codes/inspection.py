from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores
import time

__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK', 'make_maps_NN_multiK']


def infer(x, enc, K, S):
    x = NHWC2NCHW(x)
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


#########################

def eval_encoder_NN_multiK(enc, obj):      # 느림 -> train data 관련 get_x와 infer는 미리 해서 저장해두면 속도 향상 예상. 실험해보기
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    embs64_tr = infer(x_tr, enc, K=64, S=16)
    embs64_te = infer(x_te, enc, K=64, S=16)

    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
    embs32_te = infer(x_te, enc.enc, K=32, S=4)

    embs64 = embs64_tr, embs64_te
    embs32 = embs32_tr, embs32_te

    return eval_embeddings_NN_multiK(obj, embs64, embs32)


def eval_embeddings_NN_multiK(obj, embs64, embs32, NN=1):
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    det_64, seg_64 = assess_anomaly_maps(obj, maps_64)

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)
    det_32, seg_32 = assess_anomaly_maps(obj, maps_32)

    maps_sum = maps_64 + maps_32
    det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum)

    maps_mult = maps_64 * maps_32
    det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult)

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }

def make_maps_NN_multiK(enc, obj, NN=1):
    t0 = time.time()
    # from eval_encoder_NN_multiK
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    t1 = time.time()
    print("x_tr = mvtecad.get_x_standardized(obj, mode='train')- {} sec".format(t1-t0))

    x_te = mvtecad.get_x_standardized(obj, mode='test')
    t2 = time.time()
    print("x_te = mvtecad.get_x_standardized(obj, mode='test') - {} sec".format(t2-t1))
    
    embs64_tr = infer(x_tr, enc, K=64, S=16)
    t3 = time.time()
    print("embs64_tr = infer(x_tr, enc, K=64, S=16) - {} sec".format(t3-t2))

    embs64_te = infer(x_te, enc, K=64, S=16)
    t4 = time.time()
    print("embs64_te = infer(x_te, enc, K=64, S=16) - {} sec".format(t4-t3))

    # x_tr = mvtecad.get_x_standardized(obj, mode='train')
    # x_te = mvtecad.get_x_standardized(obj, mode='test')

    embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
    t5 = time.time()
    print("embs32_tr = infer(x_tr, enc.enc, K=32, S=4)- {} sec".format(t5-t4))

    embs32_te = infer(x_te, enc.enc, K=32, S=4)
    t6 = time.time()
    print("embs32_te = infer(x_te, enc.enc, K=32, S=4) - {} sec".format(t6-t5))

    embs64 = embs64_tr, embs64_te
    t7 = time.time()
    print("embs64 = embs64_tr, embs64_te - {} sec".format(t7-t6))

    embs32 = embs32_tr, embs32_te

    # from eval_embeddings_NN_multiK
    emb_tr, emb_te = embs64

    t8 = time.time()
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    t9 = time.time()
    print("maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN) - {} sec".format(t9-t8))

    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    t10 = time.time()
    print("maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16) - {} sec".format(t10-t9))

    emb_tr, emb_te = embs32
    t11 = time.time()
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)    
    t12 = time.time()
    print("maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN) - {} sec".format(t12-t11))

    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)
    t13 = time.time()
    print("maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4) - {} sec".format(t13-t12))

    maps_sum = maps_64 + maps_32
    t14 = time.time()
    print("maps_sum = maps_64 + maps_32 - {} sec".format(t14-t13))

    maps_mult = maps_64 * maps_32
    t15 = time.time()
    print("maps_mult = maps_64 * maps_32 - {} sec".format(t15-t14))

    return {
        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }

########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)  # test 각 점으로부터, train 중 가장 가까운 점까지의 거리
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
