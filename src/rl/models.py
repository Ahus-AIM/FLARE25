import torch
from torch import nn

from src.model.modeling.common import normalize
from src.model.modeling.normalized_mask_decoder3D import (
    Attention,
    MLPBlock3D,
)


class TransformerLayer(nn.Module):
    def __init__(self, attention, mlp):
        super().__init__()
        self.attention = attention
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, x, x, residual_stream=x)
        x = self.mlp(x, residual_stream=x)
        return x


class PromptAttentionNet(nn.Module):
    def __init__(self, num_layers: int, emb_dim: int, num_heads: int, output_size: int):
        """
        Attention over prompts and instances.
        """
        super().__init__()

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.output_size = output_size

        self.threshold_embedding = nn.Parameter(torch.randn(self.emb_dim))
        self.threshold_embedding.data = normalize(self.threshold_embedding.data, dim=-1)

        self.embedding_for_missing_prompts = nn.Parameter(torch.randn(self.emb_dim))
        self.embedding_for_missing_prompts.data = normalize(self.embedding_for_missing_prompts.data, dim=-1)

        self.attn_over_prompts = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLPBlock3D(emb_dim, mlp_dim=emb_dim * 2, act=nn.GELU)
            attn = Attention(emb_dim, num_heads, downsample_rate=1)
            self.attn_over_prompts.append(TransformerLayer(attn, mlp))

        self.attn_over_instances = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLPBlock3D(emb_dim, mlp_dim=emb_dim * 2, act=nn.GELU)
            attn = Attention(emb_dim, num_heads, downsample_rate=1)
            self.attn_over_instances.append(TransformerLayer(attn, mlp))

        self.projection = nn.Linear(emb_dim, output_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, emb_dim) where B is the batch size and N is the number of prompts.
            mask: (B, N) where B is the batch size and N is the number of prompts.

        Returns:
            x: (B, output_size) where B is the batch size.
        """
        B, N, emb_dim = x.shape
        # prompts that are not observed yet (mask=0), we set equal to the learnable vector
        x = x * mask.unsqueeze(-1) + (~mask).unsqueeze(-1) * self.embedding_for_missing_prompts
        expanded_threshold_embedding = self.threshold_embedding[None, None, :].expand(B, -1, -1)
        x = torch.cat([x, expanded_threshold_embedding], dim=1)  # (B, N+1, emb_dim)

        for layer in self.attn_over_prompts:
            x = layer(x)

        # x still has shape (B, N+1, emb_dim)

        x = x[:, -1, :].unsqueeze(0)  # (1, B, emb_dim)

        for layer in self.attn_over_instances:
            x = layer(x)

        x = x.squeeze(0)  # (B, emb_dim)

        x = self.projection(x)  # (B, output_size)

        return x

    def normalize_weights(self):
        self.threshold_embedding.data = normalize(self.threshold_embedding.data, dim=-1)
        self.embedding_for_missing_prompts.data = normalize(self.embedding_for_missing_prompts.data, dim=-1)
