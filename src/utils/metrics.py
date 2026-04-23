import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from src.utils.SurfaceDice import (
    compute_dice_coefficient,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)


def compute_multi_class_dsc(gt, seg, label_ids):
    present_labels = set(np.unique(gt)[1:]) & set(label_ids)
    dsc = [None] * len(present_labels)
    for idx, i in enumerate(present_labels):
        gt_i = gt == i
        seg_i = seg == i
        dsc[idx] = compute_dice_coefficient(gt_i, seg_i)

    if len(dsc) == 0:
        return np.nan

    return np.nanmean(dsc)


def compute_multi_class_nsd(gt, seg, spacing, label_ids, tolerance=2.0):
    present_labels = set(np.unique(gt)[1:]) & set(label_ids)
    nsd = [None] * len(present_labels)
    for idx, i in enumerate(present_labels):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
        nsd[idx] = compute_surface_dice_at_tolerance(surface_distance, tolerance)

    if len(nsd) == 0:
        return np.nan

    return np.nanmean(nsd)


def _label_overlap(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same number of elements")

    if not np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.int64, copy=False)
    else:
        x = x.astype(np.int64, copy=False)
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.int64, copy=False)
    else:
        y = y.astype(np.int64, copy=False)

    nx = int(x.max()) + 1 if x.size else 1
    ny = int(y.max()) + 1 if y.size else 1

    overlap = np.zeros((nx, ny), dtype=np.int64)
    np.add.at(overlap, (x, y), 1)
    return overlap


def _intersection_over_union(masks_true, masks_pred):
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    if iou.size == 0 or iou.shape[0] == 0 or iou.shape[1] == 0:
        return 0, []

    n_min = min(iou.shape[0], iou.shape[1])
    if n_min == 0:
        return 0, []

    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = int(match_ok.sum())
    matched_pairs = [(t, p) for t, p, ok in zip(true_ind, pred_ind, match_ok) if ok]
    return tp, matched_pairs


def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg > 0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        tp, matched_pairs = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        # print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0
        matched_pairs = None

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return f1_score, matched_pairs


def ccl_3d_fast(
    mask: np.ndarray,
    connectivity: int = 6,
    verbose: bool = False,
) -> np.ndarray:
    mask_np = mask.copy()

    if mask_np.dtype != bool and mask_np.dtype != np.bool_:
        mask_np = mask_np > 0

    if not np.any(mask_np):
        return np.zeros_like(mask_np, dtype=np.int32)

    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = np.ones((3, 3, 3), dtype=bool)
        struct[0, 0, 0] = struct[0, 0, 2] = struct[0, 2, 0] = struct[0, 2, 2] = False
        struct[2, 0, 0] = struct[2, 0, 2] = struct[2, 2, 0] = struct[2, 2, 2] = False
    elif connectivity == 26:
        struct = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError("connectivity must be 6, 18, or 26")

    labeled_array, num_features = ndimage.label(mask_np, structure=struct)

    if verbose:
        print(f"Found {num_features} connected components")
        if num_features > 0:
            sizes = np.bincount(labeled_array.ravel())[1:]
            print(
                f"Component sizes - min: {sizes.min()}, max: {sizes.max()}, mean: {sizes.mean():.1f}"
            )

    return labeled_array.astype(np.int32)
