import argparse
import json
import multiprocessing as mp
import shutil
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import monai.transforms as mtf
import numpy as np
import torch
from monai.data import MetaTensor
from tqdm import tqdm


def process_npz_cpu(
    path: Path,
    output_dir: Path,
    data_dir: Path,
) -> Tuple[List[int], str]:
    relative_path = path.relative_to(data_dir)
    save_path = output_dir / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if "microscopy" in str(path).lower():
        transform = mtf.Compose(
            [
                mtf.CropForegroundd(
                    keys=["image", "label"], source_key="image", allow_smaller=True
                ),
                mtf.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )
    else:
        transform = mtf.Compose(
            [
                mtf.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 1.5),
                    mode=("bilinear", "nearest"),
                ),
                mtf.CropForegroundd(
                    keys=["image", "label"], source_key="image", allow_smaller=True
                ),
                mtf.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )

    with np.load(path, allow_pickle=True) as npz:
        image_np = np.array(npz["imgs"], dtype=np.float32)
        label_np = np.array(npz["gts"], dtype=np.int32)
        spacing = npz["spacing"]

        affine = np.diag([spacing[2], spacing[1], spacing[0], 1.0])
        image_mt = MetaTensor(
            torch.from_numpy(image_np[None, ...]),
            affine=torch.from_numpy(affine),
            channel_dim=0,
        )
        label_mt = MetaTensor(
            torch.from_numpy(label_np[None, ...]),
            affine=torch.from_numpy(affine),
            channel_dim=0,
        )

        data_dict = {"image": image_mt, "label": label_mt}
        processed = transform(data_dict)

        image_processed_np = processed["image"].squeeze(0).numpy()
        label_processed_np = processed["label"].squeeze(0).numpy()

        np.savez_compressed(save_path, imgs=image_processed_np, gts=label_processed_np)

    return str(save_path)


def find_all_npz_files(data_dir: Path) -> List[Dict[str, Any]]:
    npz_files = []
    for file_path in data_dir.rglob("*.npz"):
        try:
            relative_path = file_path.relative_to(data_dir)
            modality_name = relative_path.parts[0]
            dataset_name = relative_path.parts[1]
            npz_files.append(
                {
                    "file_path": file_path,
                    "modality_name": modality_name,
                    "dataset_name": dataset_name,
                }
            )
        except IndexError:
            print(f"Warning: Skipping file with unexpected path structure: {file_path}")
    return npz_files


def generate_train_path_json(
    data_dir: Path,
    output_dir: Path,
    class_info_path: Path,
    max_workers: int,
):
    try:
        with open(class_info_path, "r") as f:
            class_info_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading class info file {class_info_path}: {e}")
        return

    num_cpus = mp.cpu_count()
    max_workers = min(max_workers, num_cpus)
    print(f"Using max_workers={max_workers} for processing.")

    print("Scanning for NPZ files...")
    all_npz_files = find_all_npz_files(data_dir)
    print(f"Found {len(all_npz_files)} NPZ files.")

    if not all_npz_files:
        print("No NPZ files found in the specified directory.")
        return

    print("Validating dataset configurations...")
    dataset_configs = {}
    invalid_datasets = set()
    for file_info in all_npz_files:
        dataset_name = file_info["dataset_name"]
        if dataset_name in dataset_configs or dataset_name in invalid_datasets:
            continue

        dataset_class_info = class_info_data.get(dataset_name)
        if not dataset_class_info or "instance_label" not in dataset_class_info:
            print(f"Warning: Invalid config for dataset '{dataset_name}'. Skipping.")
            invalid_datasets.add(dataset_name)
            continue

        dataset_configs[dataset_name] = {
            "instance_label": dataset_class_info["instance_label"],
            "prompt_valid_keys": [k for k in dataset_class_info if k.isdigit()],
        }

    valid_files = [
        f for f in all_npz_files if f["dataset_name"] not in invalid_datasets
    ]
    print(
        f"Processing {len(valid_files)} files from {len(dataset_configs)} valid datasets."
    )

    processed_files = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for file_info in valid_files:
            future = executor.submit(
                process_npz_cpu,
                file_info["file_path"],
                output_dir,
                data_dir,
            )
            future_to_info[future] = file_info

        for future in tqdm(
            as_completed(future_to_info),
            total=len(future_to_info),
            desc="Processing NPZ files on CPUs",
        ):
            file_info = future_to_info[future]
            try:
                save_path = future.result()

                dataset_key = (file_info["modality_name"], file_info["dataset_name"])
                if dataset_key not in processed_files:
                    processed_files[dataset_key] = {
                        "files": [],
                        "instance_label": dataset_configs[file_info["dataset_name"]][
                            "instance_label"
                        ],
                    }
                processed_files[dataset_key]["files"].append({"file_path": save_path})
            except Exception as e:
                print(f"Error processing file {file_info['file_path']}: {e}")

    final_data = OrderedDict()
    sorted_keys = sorted(
        processed_files.keys(), key=lambda k: (k[0].lower(), k[1].lower())
    )

    for modality_name, dataset_name in sorted_keys:
        if modality_name not in final_data:
            final_data[modality_name] = OrderedDict()

        dataset_info = processed_files[(modality_name, dataset_name)]
        dataset_info["files"].sort(key=lambda x: x["file_path"].lower())
        final_data[modality_name][dataset_name] = dataset_info
        print(
            f"Completed {modality_name}/{dataset_name}: {len(dataset_info['files'])} files processed."
        )

    total_files = sum(
        len(ds["files"]) for mod in final_data.values() for ds in mod.values()
    )
    total_datasets = sum(len(mod) for mod in final_data.values())
    total_modalities = len(final_data)

    output_dict = OrderedDict(
        [
            (
                "summary",
                {
                    "total_files": total_files,
                    "total_datasets": total_datasets,
                    "total_modalities": total_modalities,
                },
            ),
            *final_data.items(),
        ]
    )

    output_json_path = output_dir / "dataset_info.json"
    with open(output_json_path, "w") as json_file:
        json.dump(output_dict, json_file, indent=4)

    print(f"JSON file saved to {output_json_path}")
    print(
        f"Total {total_files} files processed across {total_datasets} datasets in {total_modalities} modalities."
    )


if __name__ == "__main__":
    # try:
    #     mp.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass

    parser = argparse.ArgumentParser(
        description="Generate train path JSON with preprocessing on CPUs"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Root directory of the data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save processed files and JSON. Defaults to data_dir.",
    )
    parser.add_argument(
        "--class_info_path",
        type=str,
        default="CVPR-BiomedSegFM/CVPR25_TextSegFMData_with_class.json",
        help="Path to the class info JSON file",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of worker processes to use",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    class_info_path = Path(args.class_info_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_train_path_json(
        data_dir,
        output_dir,
        class_info_path,
        args.max_workers,
    )
