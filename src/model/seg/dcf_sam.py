import torch
import torch.nn as nn

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer


class DCF_SAM(nn.Module):
    def __init__(
        self,
        text_dim,
        image_size=(128, 256, 256),
        embed_dim=768,
        feat_size=(4, 4, 2),
        pass_num=1,
        transformer_depth=2,
        mlp_dim=2048,
        num_heads=12,
        num_kv_heads=4,
    ):
        super().__init__()
        self.image_size = image_size
        self.feat_size = feat_size
        self.pass_num = pass_num

        self.mask_decoder = MaskDecoder(
            transformer_dim=embed_dim,
            feat_shape=self.feat_size,
            text_dim=text_dim,
            transformer=TwoWayTransformer(
                depth=transformer_depth,
                embedding_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            ),
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=self.feat_size,
            input_image_size=image_size,
        )

        self.text_adapt = nn.Linear(text_dim, embed_dim)

    def forward(self, image_features, text_emb):
        image_embedding = image_features[-1]
        high_res_features = image_features[:-1]

        logits, sam_tokens_out = self.rough_decoder(
            image_embedding,
            text_emb=text_emb,
            high_res_features=high_res_features,
        )

        for _ in range(self.pass_num):
            logits, sam_tokens_out = self.refine_decoder(
                image_embedding,
                text_emb=text_emb,
                masks=torch.sigmoid(logits),
                sam_tokens_out=sam_tokens_out,
                high_res_features=high_res_features,
            )

        return logits

    def rough_decoder(self, image_embedding, text_emb, high_res_features):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            self.text_adapt(text_emb)
        )

        low_res_masks, sam_tokens_out = self.mask_decoder(
            image_embedding=image_embedding,
            text_embedding=text_emb,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=high_res_features,
        )

        return low_res_masks, sam_tokens_out

    def refine_decoder(
        self, image_embedding, text_emb, masks, sam_tokens_out, high_res_features
    ):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            self.text_adapt(text_emb), masks, sam_tokens_out
        )

        low_res_masks, sam_tokens_out = self.mask_decoder(
            image_embedding=image_embedding,
            text_embedding=text_emb,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=high_res_features,
        )

        return low_res_masks, sam_tokens_out
