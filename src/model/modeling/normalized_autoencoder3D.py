import torch
from torch import Tensor, nn

from model.modeling.normalized_image_encoder3D import Block3D, NormalizedImageEncoderViT3D, normalize


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


class UpscaleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            LayerNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            LayerNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class NormalizedDecoder3D(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int, out_chans: int, input_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.out_chans = out_chans

        self.blocks = nn.ModuleList(
            [
                Block3D(
                    dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                input_size,
                input_size,
                input_size,
                embed_dim,
            )
        )

        self.output_upscaling = nn.Sequential(
            UpscaleBlock3D(embed_dim, embed_dim // 4),
            UpscaleBlock3D(embed_dim // 4, embed_dim // 8),
            UpscaleBlock3D(embed_dim // 8, embed_dim // 16),
            nn.ConvTranspose3d(embed_dim // 16, embed_dim // 16, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.GELU(),
            nn.Conv3d(embed_dim // 16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed
        x = normalize(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 4, 1, 2, 3)  # emb dim as channel
        x = self.output_upscaling(x)
        x = torch.sigmoid(x)
        return x


class NormalizedMaskedAutoencoder3D(nn.Module):
    def __init__(self, encoder: NormalizedImageEncoderViT3D, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_blocks = 5

    def encoder_forward(self, x: Tensor) -> Tensor:
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        x = normalize(x)
        x = self.mask_volume(x)

        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.neck(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = normalize(x)

        return x

    def mask_volume(self, x: Tensor) -> Tensor:
        B, _, _, _, C = x.shape
        x_flat = x.reshape(B, -1, C)
        self.indices = torch.randperm(x_flat.shape[1])[: self.num_blocks**3]
        masked_volume = x_flat[:, self.indices]
        masked_volume = masked_volume.reshape(B, self.num_blocks, self.num_blocks, self.num_blocks, C)
        return masked_volume

    def expand_volume(self, masked: torch.Tensor) -> torch.Tensor:
        B, _, _, _, C = masked.shape
        expanded = torch.zeros((B, 8 * 8 * 8, C), device=masked.device, dtype=masked.dtype)
        expanded[:, self.indices, :] = masked.reshape(B, -1, C)
        expanded = expanded.reshape(B, 8, 8, 8, C)
        return expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder_forward(x)
        emb = self.expand_volume(emb)
        x_hat = self.decoder(emb)
        # x_hat = F.interpolate(x_hat, size=x.shape[-3:], mode="trilinear", align_corners=False) # NOTE uncomment if decoder output upscaling results in lower resolution than input
        return x_hat


if __name__ == "__main__":
    input_batch = torch.randn(4, 1, 128, 128, 128)
    encoder = NormalizedImageEncoderViT3D(
        depth=6,
        embed_dim=768,
        img_size=128,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=384,
    )
    decoder = NormalizedDecoder3D(
        embed_dim=384,
        depth=1,
        num_heads=12,
        out_chans=1,
        input_size=8,
    )

    autoencoder = NormalizedMaskedAutoencoder3D(encoder, decoder)
    output = autoencoder(input_batch)
