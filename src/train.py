import os
from dataclasses import dataclass, field
from typing import List

import torch
import torch.distributed as dist
import transformers
from muon import Muon
from transformers import AutoTokenizer, Trainer

from src.dataset import BiomedSegDataset
from src.model.modeling import ESICA, ESICAConfig


def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args):
    if is_rank_zero():
        print(*args)


@dataclass
class ModelArguments:
    wb_project: str = "Text3DSAM"

    text_model: str = "bert-base-uncased"

    image_size: List[int] = field(default_factory=lambda: [128, 256, 256])
    embed_dim: int = 768
    patch_size: List[int] = field(default_factory=lambda: [64, 64, 64])
    pass_num: int = 1
    transformer_depth: int = 2
    mlp_dim: int = 2048
    num_heads: int = 12
    num_kv_heads: int = 4

    focal_weight: float = 1.0
    dice_weight: float = 1.0

    freeze_text_encoder: bool = False
    pretrained_model: str = ""


@dataclass
class DataArguments:
    data_dir: str = "CVPR-BiomedSegFM/3D_train_npz_all"
    prompt_dir: str = "CVPR-BiomedSegFM/CVPR25_TextSegFMData_with_class.json"
    max_length: int = 128

    pos_num: int = 2
    neg_num: int = 0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: bool = True

    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000

    label_names: List[str] = field(default_factory=lambda: ["labels"])

    bf16: bool = True
    output_dir: str = "./output/Text3DSAM"
    num_train_epochs: float = 30
    per_device_train_batch_size: int = 32
    # per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "no"
    # eval_accumulation_steps: int = 1
    # eval_steps: float = 0.1
    save_strategy: str = "steps"
    save_steps: float = 0.05
    save_total_limit: int = 1
    logging_steps: float = 0.001

    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8

    report_to: str = "wandb"
    run_name: str = "Training"


@dataclass
class DataCollator:
    def __init__(self):
        return

    def __call__(self, batch: List[dict]) -> dict:
        images, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch]
            for key in ("image", "input_ids", "label", "attention_mask")
        )

        images_stacked = torch.stack(images, dim=0)
        labels_stacked = torch.stack(labels, dim=0)
        input_ids_stacked = torch.stack(input_ids, dim=0)
        attention_mask_stacked = torch.stack(attention_mask, dim=0)

        batch_size, num_samples = images_stacked.shape[:2]
        total_samples = batch_size * num_samples

        images = images_stacked.view(total_samples, *images_stacked.shape[2:])
        labels = labels_stacked.view(total_samples, *labels_stacked.shape[2:])

        input_ids = input_ids_stacked.view(total_samples, -1)
        attention_mask = attention_mask_stacked.view(total_samples, -1)

        true_lengths = [int(mask.sum()) for mask in attention_mask]
        max_len_in_batch = max(true_lengths)

        trimmed_input_ids = [tensor[:max_len_in_batch] for tensor in input_ids]
        trimmed_attention_mask = [
            tensor[:max_len_in_batch] for tensor in attention_mask
        ]

        input_ids = torch.stack(trimmed_input_ids, dim=0)
        attention_mask = torch.stack(trimmed_attention_mask, dim=0)

        return {
            "image": images,
            "label": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["WANDB_PROJECT"] = model_args.wb_project

    data_args.roi = model_args.image_size

    if model_args.pretrained_model:
        model = ESICA.from_pretrained(model_args.pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_model)
    else:
        config = ESICAConfig.from_dict(vars(model_args))
        tokenizer = AutoTokenizer.from_pretrained(model_args.text_model)
        model = ESICA(config)
        model.initialize_weights_for_training()

    if model_args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False

    train_dataset = BiomedSegDataset(data_args, tokenizer)
    data_collator = DataCollator()

    rank0_print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")
    rank0_print(
        f"Text encoder parameters number: {sum(p.numel() for p in model.text_encoder.parameters()) / 1e6:.2f}M"
    )
    rank0_print(
        f"Learnable parameters number: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if training_args.use_muon:
        optimizer = Muon(
            [p for p in model.parameters() if p.requires_grad],
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )

        trainer.optimizer = optimizer

    resume_checkpoint = None
    if os.path.exists(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(training_args.output_dir, d))
        ]

        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            rank0_print(f"Resuming from checkpoint: {resume_checkpoint}")

    rank0_print(f"Using {trainer.optimizer.__class__.__name__} optimizer")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
