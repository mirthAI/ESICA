import argparse
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def process_npz(file_info, class_info_data):
    path, modality_name, dataset_name = file_info
    dataset_class_info = class_info_data.get(dataset_name)

    if dataset_class_info is None:
        return None, f"No class info found for dataset {dataset_name}"

    instance_label = dataset_class_info.get("instance_label", None)

    if instance_label is None:
        return None, f"No instance label found for dataset {dataset_name}"

    prompt_valid_keys = [k for k in dataset_class_info.keys() if k.isdigit()]

    if instance_label == 1:
        numeric_keys = [1]
    else:
        try:
            npz = np.load(path, allow_pickle=True)
            gts = np.array(npz["gts"])

            unique_values = np.unique(gts)
            numeric_keys = unique_values[unique_values > 0]

            if len(numeric_keys) == 0:
                return None, f"No foreground pixels found in: {path}"

            valid_keys = []

            for k in numeric_keys:
                class_id = int(k)
                if str(class_id) in prompt_valid_keys:
                    class_mask = gts == k
                    if np.sum(class_mask) > 0:
                        valid_keys.append(class_id)
                    else:
                        print(f"No pixels found for class {class_id} in: {path}")

            numeric_keys = valid_keys
        except Exception as e:
            return None, f"Error processing file {path}: {e}"

    if len(numeric_keys) == 0:
        return None, None

    return {
        "file_path": path,
        "numeric_class": numeric_keys,
        "modality_name": modality_name,
        "dataset_name": dataset_name,
    }, None


def collect_all_npz_files(data_dir):
    npz_files = []
    data_path = Path(data_dir)

    for npz_file in data_path.rglob("*.npz"):
        try:
            relative_path = npz_file.relative_to(data_path)
            path_parts = relative_path.parts
            if len(path_parts) >= 3:
                modality_name = path_parts[0]
                dataset_name = path_parts[1]
                npz_files.append((str(npz_file), modality_name, dataset_name))
            else:
                print(
                    f"Warning: Skipping file with unexpected path structure: {npz_file}"
                )
        except Exception as e:
            print(f"Warning: Error processing file {npz_file}: {e}")

    return npz_files


def generate_train_path_json(data_dir, output_path, class_info_path, max_workers):
    try:
        with open(class_info_path, "r") as f:
            class_info_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Class info file not found at {class_info_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {class_info_path}")
        return

    print("Collecting all NPZ files...")
    all_npz_files = collect_all_npz_files(data_dir)
    print(f"Found {len(all_npz_files)} NPZ files")

    if not all_npz_files:
        print("No NPZ files found")
        return

    data_dict = {}
    warned_datasets = set()
    print(f"Processing files with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_npz, file_info, class_info_data): file_info
            for file_info in all_npz_files
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing NPZ files"
        ):
            file_info = futures[future]
            try:
                result, error = future.result()
                if error:
                    if "No class info found for dataset" in error:
                        dataset_name = file_info[2]
                        if dataset_name not in warned_datasets:
                            print(f"Warning: {error}")
                            warned_datasets.add(dataset_name)
                    else:
                        print(f"Warning: {error}")
                    continue

                if result is None:
                    continue

                modality_name = result["modality_name"]
                dataset_name = result["dataset_name"]

                if modality_name not in data_dict:
                    data_dict[modality_name] = {}
                if dataset_name not in data_dict[modality_name]:
                    dataset_class_info = class_info_data.get(dataset_name)
                    instance_label = dataset_class_info.get("instance_label", None)
                    data_dict[modality_name][dataset_name] = {
                        "files": [],
                        "instance_label": instance_label,
                    }

                data_dict[modality_name][dataset_name]["files"].append(
                    {
                        "file_path": result["file_path"],
                        "numeric_class": result["numeric_class"],
                    }
                )
            except Exception as e:
                print(f"Error processing file {file_info[0]}: {e}")

    sorted_data_dict = OrderedDict()

    for modality_name in sorted(data_dict.keys(), key=str.lower):
        sorted_datasets = OrderedDict()
        for dataset_name in sorted(data_dict[modality_name].keys(), key=str.lower):
            dataset_data = data_dict[modality_name][dataset_name]
            dataset_data["files"].sort(key=lambda x: x["file_path"].lower())
            sorted_datasets[dataset_name] = {
                "files": dataset_data["files"],
                "instance_label": dataset_data["instance_label"],
            }
        sorted_data_dict[modality_name] = sorted_datasets

    total_files = sum(
        len(dataset_data["files"])
        for modality in data_dict.values()
        for dataset_data in modality.values()
    )
    total_datasets = sum(len(modality) for modality in data_dict.values())
    total_modalities = len(data_dict)
    output_dict = OrderedDict()
    output_dict["summary"] = {
        "total_files": total_files,
        "total_datasets": total_datasets,
        "total_modalities": total_modalities,
        "files_per_modality": {},
    }

    for modality, datasets in sorted_data_dict.items():
        modality_file_count = sum(
            len(dataset_data["files"]) for dataset_data in datasets.values()
        )
        output_dict["summary"]["files_per_modality"][modality] = modality_file_count
        output_dict[modality] = datasets

    output_json_path = os.path.join(output_path, "dataset_info.json")

    with open(output_json_path, "w") as json_file:
        json.dump(output_dict, json_file, indent=4)

    print(f"JSON file saved to {output_json_path}")
    print(
        f"Total {total_files} files found across {total_datasets} datasets in {total_modalities} modalities in {data_dir}"
    )
    print("Files per modality:")

    for modality, count in output_dict["summary"]["files_per_modality"].items():
        print(f"  {modality}: {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train path JSON")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument(
        "--class_info_path",
        type=str,
        default="CVPR-BiomedSegFM/CVPR25_TextSegFMData_with_class.json",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    output_path = args.data_dir if args.output_path is None else args.output_path
    class_info_path = args.class_info_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    generate_train_path_json(data_dir, output_path, class_info_path, args.max_workers)
