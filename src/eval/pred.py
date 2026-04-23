import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import monai.transforms as mtf
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from transformers import AutoTokenizer

from src.model.modeling import Med3DSeg


def worker_process(gpu_id, files, model_path, input_dir, output_dir):
    device = torch.device(f"cuda:{gpu_id}")

    model = Med3DSeg.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_model)
    model.eval()
    model = torch.compile(model)

    pre_transforms = mtf.Compose(
        [
            mtf.ToTensord(keys="image", device=device, dtype=torch.float32),
            mtf.Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5)),
            mtf.CropForegroundd(
                keys="image",
                source_key="image",
                return_transform=True,
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

    no_spacing_pre_transforms = mtf.Compose(
        [
            mtf.ToTensord(keys="image", device=device, dtype=torch.float32),
            mtf.CropForegroundd(
                keys="image",
                source_key="image",
                return_transform=True,
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

    post_transforms = mtf.Compose(
        [
            mtf.EnsureTyped(keys="pred", dtype=torch.float32),
            mtf.EnsureChannelFirstd(keys="pred", channel_dim="no_channel"),
            mtf.Invertd(
                keys="pred",
                transform=pre_transforms,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
        ]
    )

    no_spacing_post_transforms = mtf.Compose(
        [
            mtf.EnsureTyped(keys="pred", dtype=torch.float32),
            mtf.EnsureChannelFirstd(keys="pred", channel_dim="no_channel"),
            mtf.Invertd(
                keys="pred",
                transform=no_spacing_pre_transforms,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
        ]
    )

    pbar = tqdm(files, desc=f"[GPU {gpu_id}]", position=gpu_id, leave=True)
    for file in pbar:
        try:
            image_path = os.path.join(input_dir, file)
            npz = np.load(image_path, allow_pickle=True)

            image_np = npz["imgs"]
            spacing = npz["spacing"]

            affine = np.diag([spacing[2], spacing[1], spacing[0], 1.0])
            image_mt = MetaTensor(
                torch.from_numpy(image_np[None, ...]),
                affine=torch.from_numpy(affine),
                channel_dim=0,
            )

            text_prompts = npz["text_prompts"].item()
            instance_label = text_prompts["instance_label"]

            data_dict = {"image": image_mt}

            original_image_shape = image_np.shape

            if "microscopy" in model_path.lower():
                data_dict = no_spacing_pre_transforms(data_dict)
                current_post_transforms = no_spacing_post_transforms
            else:
                data_dict = pre_transforms(data_dict)
                current_post_transforms = post_transforms

            preprocessed_image = data_dict["image"].unsqueeze(0)

            segs_probs = []
            class_ids_in_order = []

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for k, v in text_prompts.items():
                    if k == "instance_label":
                        continue

                    class_id = int(k)
                    class_ids_in_order.append(class_id)
                    prompt_text = v

                    prompt_tokens = tokenizer(
                        prompt_text,
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(device)

                    input_ids = prompt_tokens["input_ids"]
                    attention_mask = prompt_tokens["attention_mask"]

                    logits = sliding_window_inference(
                        inputs=preprocessed_image,
                        roi_size=model.config.image_size,
                        sw_batch_size=64,
                        predictor=lambda x: model.generate_batch(
                            image=x,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        ),
                        overlap=0.5,
                        mode="gaussian",
                    )

                    pred_prob = torch.sigmoid(logits)

                    if instance_label == 0:
                        segs_probs.append(pred_prob.squeeze(0).squeeze(0).detach())
                    else:
                        segs_probs = (
                            (pred_prob > 0.5).float().squeeze(0).squeeze(0).detach()
                        )
                        break

            if instance_label == 0:
                stacked_probs_fg = torch.stack(segs_probs, dim=0)
                max_fg_prob, _ = torch.max(stacked_probs_fg, dim=0)
                prob_background = torch.clamp(1.0 - max_fg_prob, min=0.0, max=1.0)
                prob_background = prob_background.unsqueeze(0)

                stacked_probs_all = torch.cat(
                    [prob_background, stacked_probs_fg], dim=0
                )
                pred_indices = torch.argmax(stacked_probs_all, dim=0)

                pred_labels = torch.zeros_like(pred_indices, dtype=torch.long)
                for index_in_stack, class_id_val in enumerate(class_ids_in_order):
                    argmax_index = index_in_stack + 1
                    pred_labels[pred_indices == argmax_index] = class_id_val

                data_dict["pred"] = pred_labels.to(device)
            else:
                data_dict["pred"] = segs_probs.to(device)

            data_dict = current_post_transforms(data_dict)
            final_labels_np = (
                data_dict["pred"].squeeze(0).cpu().numpy().astype(np.uint8)
            )

            assert (
                final_labels_np.shape == original_image_shape
            ), f"Shape mismatch: original {original_image_shape} vs final {final_labels_np.shape}"

            base_filename = os.path.basename(image_path)
            output_filename = os.path.join(output_dir, base_filename)
            np.savez_compressed(output_filename, segs=final_labels_np)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {file}: {str(e)}")


def pred_multi_gpu(args, global_start_time):
    available_gpus = torch.cuda.device_count()
    max_workers = (
        min(available_gpus, args.max_workers)
        if args.max_workers is not None
        else available_gpus
    )

    print(f"Using {available_gpus} GPUs with ProcessPoolExecutor")
    print(f"Max workers: {max_workers}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError("No .npz files found in the input directory.")

    print(f"Found {len(npz_files)} files to process")

    file_splits = [npz_files[i::max_workers] for i in range(max_workers)]

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(
                worker_process,
                gpu_id,
                file_splits[gpu_id],
                args.model_path,
                args.input_dir,
                args.output_dir,
            )
            for gpu_id in range(max_workers)
        ]
        for f in futures:
            f.result()

    global_end_time = time.time()
    print(f"Total time taken: {global_end_time - global_start_time:.2f} seconds")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    global_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/Med3DSeg")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/eval/preds")
    parser.add_argument("--max_workers", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, exiting...")
        exit(1)
    elif torch.cuda.device_count() == 1:
        print("Only 1 GPU available")

    pred_multi_gpu(args, global_start_time)
