import math
from typing import List, Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def normalize(x: Tensor, dim: int = -1) -> Tensor:
    res: torch.Tensor = x / x.norm(p=2, dim=dim, keepdim=True).clamp(min=1e-3)
    return res


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MLPBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lin_up = nn.Linear(embedding_dim, 2 * mlp_dim)
        self.lin_down = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

        self.mlp_alpha_init_value: float = 0.1
        self.mlp_alpha_init_scaling: float = 1.0
        self.mlp_alpha: torch.nn.Parameter = torch.nn.Parameter(
            self.mlp_alpha_init_scaling * torch.ones(embedding_dim, dtype=torch.float32)
        )

        self.suv_init_value: float = 1.0
        self.suv_init_scaling: float = 1.0
        self.suv: torch.nn.Parameter = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(2 * mlp_dim, dtype=torch.float32)
        )

    def forward(self, x: Tensor, residual_stream: Tensor) -> Tensor:
        # Apply MLP
        uv = self.lin_up(x)
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.embedding_dim**0.5))
        uv = suv * uv
        u, v = uv.chunk(2, dim=-1)
        x_mlp = u * self.act(v)
        x_mlp = self.lin_down(x_mlp)

        # Step on the hypersphere
        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        residual_stream = normalize(residual_stream, dim=-1)
        x_mlp = normalize(x_mlp, dim=-1)

        out = normalize(residual_stream + lr * (x_mlp - residual_stream), dim=-1)

        return out

    def normalize_weights(self):
        # print("Normalizing MLP block")
        self.lin_up.weight.data.copy_(normalize(self.lin_up.weight.data, dim=1))
        self.lin_down.weight.data.copy_(normalize(self.lin_down.weight.data, dim=0))


class NormalizedTwoWayTransformer3D(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock3D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, x, y, z = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe

        queries = self.final_attn_token_to_image(q=q, k=k, v=keys, residual_stream=queries)

        return queries, keys


class TwoWayAttentionBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)

        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # Self attention block
        # if self.skip_first_layer_pe:
        #     queries = self.self_attn(q=queries, k=queries, v=queries, residual_stream=None)
        #     # NOTE: There is no residual connection here, as there is no residual connection in SAM, SAM2 or SAM-Med3D
        #     # see https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/transformer.py#L156
        #     # In the SAM paper, it is stated that "Each self/cross-attention and MLP has a residual connection".
        #     # Is this a mistake in the implementation or in the paper?
        # else:
        #     q = queries + query_pe
        #     queries = self.self_attn(q=q, k=q, v=queries, residual_stream=queries)

        queries = self.self_attn(q=queries + query_pe, k=queries + query_pe, v=queries, residual_stream=queries)

        # Cross attention block, tokens attending to image embedding
        # q = queries + query_pe
        # k = keys + key_pe
        queries = self.cross_attn_token_to_image(q=queries + query_pe, k=keys + key_pe, v=keys, residual_stream=queries)

        # MLP block
        queries = self.mlp(queries, residual_stream=queries)

        # Cross attention block, image embedding attending to tokens
        # q = queries + query_pe
        # k = keys + key_pe
        keys = self.cross_attn_image_to_token(q=keys + key_pe, k=queries + query_pe, v=queries, residual_stream=keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.attn_alpha_init_value: float = 0.1
        self.attn_alpha_init_scaling: float = 1.0
        self.attn_alpha: torch.nn.Parameter = torch.nn.Parameter(
            self.attn_alpha_init_scaling * torch.ones(embedding_dim, dtype=torch.float32)
        )

        self.sqk_init_value: float = 1.0
        self.sqk_init_scaling: float = 1.0
        self.sqk: torch.nn.Parameter = torch.nn.Parameter(
            self.sqk_init_scaling * torch.ones(self.internal_dim, dtype=torch.float32)
        )

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, residual_stream: Tensor | None) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, self.num_heads, 1, self.internal_dim // self.num_heads
        )

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        q = sqk * normalize(q, dim=-1)
        k = sqk * normalize(k, dim=-1)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn * math.sqrt(c_per_head)  # sqrt(d_k) instead of 1/sqrt(d_k) as q and k are normalized.
        attn = torch.softmax(attn, dim=-1)

        # Get output
        attn_out = attn @ v
        attn_out = self._recombine_heads(attn_out)
        attn_out = self.out_proj(attn_out)

        out = normalize(attn_out, dim=-1)
        if residual_stream is not None:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            out = normalize(residual_stream + lr * (attn_out - residual_stream), dim=-1)

        return out

    def normalize_weights(self):
        self.q_proj.weight.data.copy_(normalize(self.q_proj.weight.data, dim=1))
        self.k_proj.weight.data.copy_(normalize(self.k_proj.weight.data, dim=1))
        self.v_proj.weight.data.copy_(normalize(self.v_proj.weight.data, dim=1))
        self.out_proj.weight.data.copy_(normalize(self.out_proj.weight.data, dim=0))


class NormalizedMaskDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        # transformer: nn.Module ,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        # self.transformer = transformer
        self.transformer = NormalizedTwoWayTransformer3D(
            depth=2,
            embedding_dim=self.transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm3d(
                transformer_dim // 4
            ),  # NOTE how strict should we be on the "nGPT" architecture? It does not actually concern convolutions
            activation(),
            nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe
        b, c, x, y, z = src.shape

        # Run the transformer
        # import IPython; IPython.embed()
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, x, y, z)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, x, y, z = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks - 4, iou_pred

    def normalize_weights(self):
        self.iou_token.weight.data.copy_(normalize(self.iou_token.weight.data, dim=-1))
        self.mask_tokens.weight.data.copy_(normalize(self.mask_tokens.weight.data, dim=-1))


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
