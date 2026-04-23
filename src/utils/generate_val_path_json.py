import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from tqdm import tqdm

val_path = "CVPR-BiomedSegFM/3D_val_npz"
gt_path = "CVPR-BiomedSegFM/3D_val_gt/3D_val_gt_text"
output_path = "CVPR-BiomedSegFM"


def scan_directory(directory_path, extension=".npz"):
    """Scan directory for files with given extension"""
    file_list = []
    for file in os.scandir(directory_path):
        if file.name.endswith(extension):
            file_list.append(file.path)
    return file_list


def process_file_pair(val_file, gt_file):
    """Process a single validation-ground truth file pair"""
    val_file_name = os.path.basename(val_file)
    gt_file_name = os.path.basename(gt_file)

    if val_file_name != gt_file_name:
        raise ValueError(f"File names do not match: {val_file_name} vs {gt_file_name}")

    val_data = np.load(val_file, allow_pickle=True)
    prompts = val_data["text_prompts"].item()
    gt_data = np.load(gt_file, allow_pickle=True)

    class_labels = np.unique(gt_data["gts"])

    file_data_list = []

    if len(prompts) == 2:
        try:
            file_data_list.append(
                {
                    "file_path": val_file,
                    "gt_path": gt_file,
                    "class_id": int(1),
                    "text_prompt": prompts["1"],
                    "only_one_prompt": True,
                }
            )
        except KeyError:
            file_data_list.append(
                {
                    "file_path": val_file,
                    "gt_path": gt_file,
                    "class_id": int(1),
                    "text_prompt": prompts["2"],
                    "only_one_prompt": True,
                }
            )
        return file_data_list, []

    warnings = []
    for class_label in class_labels:
        if class_label == 0:
            continue

        if str(class_label) not in prompts:
            warnings.append(
                f"Warning: Class {class_label} not found in prompts. file: {val_file_name}"
            )
            continue

        if class_label not in gt_data["gts"]:
            warnings.append(
                f"Warning: Class {class_label} not found in gts. file: {val_file_name}"
            )
            continue

        file_data_list.append(
            {
                "file_path": val_file,
                "gt_path": gt_file,
                "class_id": int(class_label),
                "text_prompt": prompts[str(class_label)],
                "only_one_prompt": False,
            }
        )

    return file_data_list, warnings


# Use multithreading for directory scanning
print("Scanning directories...")
with ThreadPoolExecutor(max_workers=2) as executor:
    val_future = executor.submit(scan_directory, val_path)
    gt_future = executor.submit(scan_directory, gt_path)

    val_list = val_future.result()
    gt_list = gt_future.result()

assert len(val_list) == len(
    gt_list
), "Mismatch in number of validation and ground truth files."

val_list = sorted(val_list)
gt_list = sorted(gt_list)

# Process files in parallel
data_list = []
all_warnings = []
lock = Lock()

print(f"Processing {len(val_list)} file pairs...")
with ThreadPoolExecutor(max_workers=8) as executor:
    # Submit all tasks
    future_to_files = {
        executor.submit(process_file_pair, val_file, gt_file): (val_file, gt_file)
        for val_file, gt_file in zip(val_list, gt_list)
    }

    # Process completed tasks with progress bar
    for future in tqdm(as_completed(future_to_files), total=len(future_to_files)):
        try:
            file_data_list, warnings = future.result()
            with lock:
                data_list.extend(file_data_list)
                all_warnings.extend(warnings)
        except Exception as e:
            val_file, gt_file = future_to_files[future]
            print(f"Error processing {val_file}: {e}")

# Print all warnings
for warning in all_warnings:
    print(warning)

data_list = sorted(data_list, key=lambda x: x["file_path"])

json_path = os.path.join(output_path, "val_data.json")
with open(json_path, "w") as f:
    json.dump(data_list, f, indent=4)
print(f"Saved val_data.json to {json_path}")
