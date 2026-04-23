from typing import List

import torch
import torch.nn.functional as F
from monai.losses import DiceFocalLoss
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from .seg.dcf_sam import DCF_SAM
from .seg.dcformer import DecompModel


class ESICAConfig(PretrainedConfig):
    model_type = "med3d_seg"

    def __init__(
        self,
        image_size=(128, 256, 256),
        patch_size=(64, 64, 64),
        embed_dim=768,
        text_model="bert-base-uncased",
        pass_num=1,
        transformer_depth=2,
        mlp_dim=2048,
        num_heads=12,
        num_kv_heads=4,
        focal_weight=1.0,
        dice_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.text_model = text_model
        self.pass_num = pass_num
        self.transformer_depth = transformer_depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight


class ESICA(PreTrainedModel):
    config_class = ESICAConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.feat_size = (
            self.config.image_size[0] // self.config.patch_size[0],
            self.config.image_size[1] // self.config.patch_size[1],
            self.config.image_size[2] // self.config.patch_size[2],
        )

        self.loss_fn = DiceFocalLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            lambda_dice=config.dice_weight,
            lambda_focal=config.focal_weight,
        )

        self.image_encoder = DecompModel(
            input_size=config.image_size,
            num_blocks=[1, 2, 3, 6, 4],
            channels=[48, 96, 192, 384, 768],
            block_types=["C", "C", "C", "T"],
        )
        self.text_config = AutoConfig.from_pretrained(config.text_model)
        self.text_encoder = AutoModel.from_config(self.text_config)

        try:
            text_dim = self.text_encoder.config.hidden_size
        except AttributeError:
            text_dim = self.text_encoder.config.dim

        self.seg_decoder = DCF_SAM(
            embed_dim=config.embed_dim,
            feat_size=self.feat_size,
            pass_num=config.pass_num,
            transformer_depth=config.transformer_depth,
            mlp_dim=config.mlp_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            text_dim=text_dim,
        )

    def initialize_weights_for_training(self):
        if (
            hasattr(self.config, "pretrained_image_encoder")
            and self.config.pretrained_image_encoder
        ):
            self.image_encoder.load_state_dict(
                load_file(self.config.pretrained_image_encoder), strict=True
            )
            print(
                f"Load pretrained image encoder from {self.config.pretrained_image_encoder}"
            )

        if (
            hasattr(self.config, "pretrained_text_encoder")
            and self.config.pretrained_text_encoder
        ):
            self.text_encoder.load_state_dict(
                load_file(self.config.pretrained_text_encoder), strict=True
            )
            print(
                f"Load pretrained text encoder from {self.config.pretrained_text_encoder}"
            )
        else:
            self.text_encoder.from_pretrained(
                self.config.text_model,
            )
            print(f"Load pretrained text encoder from {self.config.text_model}")

    def forward(self, image, label, input_ids, attention_mask):
        text_feature = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = text_feature.last_hidden_state[:, 0].to(dtype=image.dtype)

        b, c, d, h, w = image.shape
        image_features = self.image_encoder(image)

        logits = self.seg_decoder(image_features, text_embeddings)

        if logits.shape[-3:] != (d, h, w):
            logits = F.interpolate(
                logits, size=(d, h, w), mode="trilinear", align_corners=False
            )

        loss = self.loss_fn(logits, label)

        return {
            "loss": loss,
            "logits": logits,
        }

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        text_feature = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = text_feature.last_hidden_state[:, 0].to(dtype=image.dtype)

        image_features = self.image_encoder(image)

        logits = self.seg_decoder(image_features, text_embeddings)

        return logits

    @torch.no_grad()
    def generate_batch(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        B = image.shape[0]

        input_ids = input_ids.repeat(B, 1)
        attention_mask = attention_mask.repeat(B, 1)

        text_feature = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = text_feature.last_hidden_state[:, 0].to(dtype=image.dtype)

        image_features = self.image_encoder(image)

        logits = self.seg_decoder(image_features, text_embeddings)

        return logits


AutoConfig.register("med3d_seg", ESICAConfig)
AutoModel.register(ESICAConfig, ESICA)
