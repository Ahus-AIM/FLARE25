from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer


def normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalizes the 2-norm of the input tensor along the specified dimension.
    """
    return x / x.norm(2, dim=dim, keepdim=True).clamp(min=1e-6)


class NormLayer3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *_ = x.shape
        x = normalize(x, dim=1)
        return x * c**0.5


def aniso_kernel(scale: tuple | list):
    """
    A helper function to compute kernel_size, padding and stride for the given scale

    Args:
        scale: scale from a current scale level
    """
    kernel_size = [3 if scale[k] > 1 else 1 for k in range(len(scale))]
    padding = [k // 2 for k in kernel_size]
    return kernel_size, padding, scale


class SegResBlock(nn.Module):
    """
    Residual network block used SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: tuple | int = 3,
        act: tuple | str = "relu",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """
        super().__init__()

        if isinstance(kernel_size, (tuple, list)):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2  # type: ignore

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act1 = get_act_layer(act)
        self.conv1 = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act2 = get_act_layer(act)
        self.conv2 = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        x += identity
        return x
