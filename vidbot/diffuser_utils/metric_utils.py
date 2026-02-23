import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def calcucate_heatmap_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12):
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def calcucate_heatmap_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12):
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)


def calcucate_heatmap_nss(pred: np.ndarray, gt: np.ndarray):

    std = np.std(pred)
    u = np.mean(pred)
    smap = (pred - u) / std
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map[fixation_map < 0.1] = 0  # binary mask
    nss = smap * fixation_map
    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)
    return nss


def calculate_points_nss(pred_uv, gt_map):
    """_summary_

    Parameters
    ----------
    pred_uv : predicted points coords with shape (N, 2)
    gt_map : groundtruth binary mask map with shape (H, W)
    """
    h, w = gt_map.shape
    gt_map = (gt_map > 0).astype(np.float32).copy()  # Between 0 and 1
    num_pred = pred_uv.shape[0]
    pred_u, pred_v = pred_uv[:, 0], pred_uv[:, 1]
    pred_u = np.clip(pred_u, 0, w - 1).astype(np.int32)
    pred_v = np.clip(pred_v, 0, h - 1).astype(np.int32)
    pred_map = gt_map[pred_v, pred_u]  # (N,)
    nss = pred_map.sum() / num_pred
    return nss


def calculated_points_dist2mask(pred_uv, gt_map):
    """_summary_

    Parameters
    ----------
    pred_uv : predicted points coords with shape (N, 2)
    gt_map : groundtruth binary mask map with shape (H, W)
    """
    h, w = gt_map.shape
    scaling_factor = np.array([w, h])  # (2,)
    gt_map = (gt_map > 0).astype(np.float32)  # Between 0 and 1
    gt_uv = np.stack(np.where(gt_map), axis=1)[:, [1, 0]]  # (M, 2)
    gt_uv_norm = gt_uv / scaling_factor[None]  # Normalize to [0, 1]
    pred_uv_norm = pred_uv / scaling_factor[None]  # Normalize to [0, 1]
    gt_uv_norm = gt_uv_norm.clip(0, 1)
    pred_uv_norm = pred_uv_norm.clip(0, 1)
    try:
        uv_dist = pairwise_distances(pred_uv_norm, gt_uv_norm)  # (N, M)
        uv_dist = np.min(uv_dist, axis=1)  # (N,)
        uv_dist = uv_dist.mean()
    except:
        print("Error in calculating distance")
        uv_dist = 0.0
    return uv_dist
