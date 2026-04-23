import argparse
import gc
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from skimage import segmentation
from tqdm import tqdm

from src.utils.metrics import (
    ccl_3d_fast,
    compute_multi_class_dsc,
    compute_multi_class_nsd,
    eval_tp_fp_fn,
)
from src.utils.SurfaceDice import compute_dice_coefficient

join = os.path.join

test_img_path = "CVPR-BiomedSegFM/3D_val_npz"
validation_gts_path = "CVPR-BiomedSegFM/3D_val_gt/3D_val_gt_text"


def do_metric(case, input_dir):
    try:
        gt_path = join(validation_gts_path, case)
        seg_path = join(input_dir, case)
        img_path = join(test_img_path, case)

        gt_npz = np.load(gt_path, allow_pickle=True)["gts"].astype(np.uint8)
        seg_npz = np.load(seg_path, allow_pickle=True)["segs"].astype(np.uint8)
        img_npz = np.load(img_path, allow_pickle=True)

        spacing = img_npz["spacing"]
        instance_label = img_npz["text_prompts"].item()["instance_label"]

        class_ids = sorted(
            [int(k) for k in img_npz["text_prompts"].item() if k != "instance_label"]
        )
        class_ids_array = np.array(class_ids, dtype=np.int32)

        if instance_label == 0:
            dsc = compute_multi_class_dsc(gt_npz, seg_npz, class_ids_array)
            nsd = compute_multi_class_nsd(gt_npz, seg_npz, spacing, class_ids_array)
            f1_score = np.nan
            dsc_tp = np.nan
        elif instance_label == 1:
            if len(np.unique(seg_npz)) == 2:
                tumor_inst = ccl_3d_fast(seg_npz, connectivity=6)

                mask = tumor_inst > 0
                max_seg = np.max(seg_npz)
                seg_npz = seg_npz.copy()
                seg_npz[mask] = tumor_inst[mask] + max_seg

            gt_npz = segmentation.relabel_sequential(gt_npz)[0]
            seg_npz = segmentation.relabel_sequential(seg_npz)[0]

            f1_score, matched_pairs = eval_tp_fp_fn(gt_npz, seg_npz)

            if matched_pairs:
                dsc_list = []
                for gt_idx, pred_idx in matched_pairs:
                    gt_mask = gt_npz == (gt_idx + 1)
                    pred_mask = seg_npz == (pred_idx + 1)
                    dsc_value = compute_dice_coefficient(gt_mask, pred_mask)
                    dsc_list.append(dsc_value)
                dsc_tp = np.mean(dsc_list)
            else:
                dsc_tp = 0

            dsc = None
            nsd = None

    except Exception as e:
        try:
            del gt_npz, seg_npz, img_npz
        except:
            pass
        gc.collect()

        print(f"Error processing case {case}: {e}")
        dsc, nsd, f1_score, dsc_tp = None, None, None, None

    # print(f"{dsc=}, {nsd=}, {f1_score=}, {dsc_tp=}")
    del gt_npz, seg_npz, img_npz
    gc.collect()

    return case, dsc, nsd, f1_score, dsc_tp


def process_with_progress(cases, input_dir, max_workers):
    print(f"Using {max_workers} workers for CPU processing.")

    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(do_metric, case, input_dir): case for case in cases
        }

        with tqdm(total=len(cases), desc="Processing metrics on CPU") as pbar:
            for future in as_completed(future_to_case):
                case, dsc, nsd, f1_score, dsc_tp = future.result()
                results[case] = (dsc, nsd, f1_score, dsc_tp)
                pbar.update(1)

    return results


def main(args):
    input_dir = args.input_dir

    test_cases = sorted(os.listdir(test_img_path))
    print(f"Total cases to process: {len(test_cases)}")

    try:
        results = process_with_progress(test_cases, input_dir, args.max_workers)

        metric = OrderedDict()
        metric["CaseName"] = []
        metric["Modality"] = []
        metric["DSC"] = []
        metric["NSD"] = []
        metric["F1"] = []
        metric["DSC_TP"] = []

        for case in test_cases:
            if case in results:
                dsc, nsd, f1_score, dsc_tp = results[case]
                modality = case.split("_")[0]

                metric["CaseName"].append(case)
                metric["Modality"].append(modality)
                metric["DSC"].append(round(dsc, 4) if dsc is not None else np.nan)
                metric["NSD"].append(round(nsd, 4) if nsd is not None else np.nan)
                metric["F1"].append(
                    round(f1_score, 4) if f1_score is not None else np.nan
                )
                metric["DSC_TP"].append(
                    round(dsc_tp, 4) if dsc_tp is not None else np.nan
                )

        metric_df = pd.DataFrame(metric)
        csv_path = join(input_dir, "metrics_results.csv")
        metric_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

        modality_stats = metric_df.groupby("Modality")[
            ["DSC", "NSD", "F1", "DSC_TP"]
        ].mean()
        overall_dsc_mean = metric_df["DSC"].mean()
        overall_nsd_mean = metric_df["NSD"].mean()
        overall_f1_mean = metric_df["F1"].mean()
        overall_dsc_tp_mean = metric_df["DSC_TP"].mean()

        print("\nModality-wise means:")
        print(modality_stats)
        print(f"\nOverall DSC mean: {overall_dsc_mean:.4f}")
        print(f"Overall NSD mean: {overall_nsd_mean:.4f}")
        print(f"Overall F1 mean: {overall_f1_mean:.4f}")
        print(f"Overall DSC_TP mean: {overall_dsc_tp_mean:.4f}")

    except Exception as e:
        print(f"Error in main processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    main(args)
