import json
import os
import random

import monai.transforms as mtf
import numpy as np
import torch
from monai.data import Dataset


class BiomedSegDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.max_length = args.max_length
        self.data_dir = args.data_dir
        self.prompt_dir = args.prompt_dir

        self.data_list = self._load_and_filter_data_info()

        with open(self.prompt_dir, "r") as f:
            self.prompt_data = json.load(f)

        self.transform = mtf.Compose(
            [
                mtf.EnsureChannelFirstd(
                    keys=["image", "label"], channel_dim="no_channel"
                ),
                mtf.SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=args.roi,
                    mode="constant",
                ),
                mtf.FgBgToIndicesd(keys="label", fg_postfix="_fg", bg_postfix="_bg"),
                mtf.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.roi,
                    pos=args.pos_num,
                    neg=args.neg_num,
                    num_samples=args.pos_num + args.neg_num,
                    fg_indices_key="label_fg",
                    bg_indices_key="label_bg",
                    allow_smaller=True,
                ),
                mtf.ToTensord(keys=["image", "label"]),
            ]
        )

    def _load_and_filter_data_info(self):
        data_info_path = os.path.join(self.data_dir, "dataset_info.json")
        with open(data_info_path, "r") as f:
            data_info = json.load(f)

        filtered_data_info = {}
        for modality, datasets in data_info.items():
            if modality == "summary":
                continue

            filtered_datasets = {}
            for dataset_name, dataset_details in datasets.items():
                files = dataset_details.get("files", [])
                if not files:
                    continue

                valid_files = [
                    f
                    for f in files
                    if f.get("numeric_class") and len(f["numeric_class"]) > 0
                ]

                if valid_files:
                    filtered_datasets[dataset_name] = {
                        "files": valid_files,
                        "instance_label": dataset_details.get("instance_label"),
                    }

            if filtered_datasets:
                filtered_data_info[modality] = filtered_datasets

        data_list = []
        for modality, datasets in filtered_data_info.items():
            for dataset_name, dataset_details in datasets.items():
                instance_label = dataset_details["instance_label"]

                for file_info in dataset_details["files"]:
                    file_path = file_info["file_path"]
                    numeric_classes = file_info["numeric_class"]

                    if instance_label == 1:
                        data_list.append(
                            {
                                "modality": modality,
                                "dataset_name": dataset_name,
                                "file_path": file_path,
                                "class": 1,
                                "instance_label": 1,
                            }
                        )
                    else:
                        for k in numeric_classes:
                            if k == 0:
                                continue
                            data_list.append(
                                {
                                    "modality": modality,
                                    "dataset_name": dataset_name,
                                    "file_path": file_path,
                                    "class": k,
                                    "instance_label": 0,
                                }
                            )

        random.seed(42)
        random.shuffle(data_list)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        dataset_name = data["dataset_name"]
        image_path = data["file_path"]
        class_id = data["class"]
        instance_label = data["instance_label"]

        image_data = np.load(image_path, allow_pickle=True)

        image = image_data["imgs"]
        label = image_data["gts"]

        if instance_label == 1:
            label = label > 0
        else:
            label = label == class_id

        if np.unique(label).size == 1:
            print(
                f"Warning: label in {image_path} for class {class_id} has no positive voxels."
            )

        transformed = self.transform({"image": image, "label": label})

        image = []
        label = []
        for t in transformed:
            image.append(t["image"])
            label.append(t["label"])

        image = torch.stack(image, dim=0)
        label = torch.stack(label, dim=0)

        dataset_prompts = self.prompt_data.get(dataset_name, {})
        class_prompts = dataset_prompts.get(str(class_id), [])

        num_crops = image.size(0)
        input_ids_list = []
        attention_mask_list = []

        for i in range(num_crops):
            prompt = random.choice(class_prompts)
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids_list.append(prompt_tokens["input_ids"][0])
            attention_mask_list.append(prompt_tokens["attention_mask"][0])

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        result = {
            "image": image,
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return result
