# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.utils import UpsampleMode, has_option

from .common import SegResBlock, aniso_kernel

__all__ = ["SegResNetDS2"]


def scales_for_resolution(resolution: tuple | list, n_stages: int | None = None):
    """
    A helper function to compute a schedule of scale at different downsampling levels,
    given the input resolution.

    .. code-block:: python

        scales_for_resolution(resolution=[1,1,5], n_stages=5)

    Args:
        resolution: input image resolution (in mm)
        n_stages: optionally the number of stages of the network
    """

    ndim = len(resolution)
    res = np.array(resolution)
    if not all(res > 0):
        raise ValueError("Resolution must be positive")

    nl = np.floor(np.log2(np.max(res) / res)).astype(np.int32)
    scales = [tuple(np.where(2**i >= 2**nl, 1, 2)) for i in range(max(nl))]
    if n_stages and n_stages > max(nl):
        scales = scales + [(2,) * ndim] * (n_stages - max(nl))
    else:
        scales = scales[:n_stages]
    return scales


class SegResEncoder(nn.Module):
    """
    SegResEncoder based on the econder structure in `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``BATCH``.
        blocks_down: number of downsample blocks in each layer. Defaults to ``[1,2,2,4]``.
        head_module: optional callable module to apply to the final features.
        anisotropic_scales: optional list of scale for each scale level.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        act: tuple | str = "relu",
        norm: tuple | str = "instance",
        blocks_down: tuple = (1, 2, 2, 4),
        head_module: nn.Module | None = None,
        anisotropic_scales: tuple | None = None,
    ):
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("`spatial_dims` can only be 1, 2 or 3.")

        # ensure normalization has affine trainable parameters (if not specified)
        norm = split_args(norm)
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)  # type: ignore

        # ensure activation is inplace (if not specified)
        act = split_args(act)
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)  # type: ignore

        filters = init_filters  # base number of features

        kernel_size, padding, _ = aniso_kernel(anisotropic_scales[0]) if anisotropic_scales else (3, 1, 1)
        self.conv_init = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            bias=False,
        )
        self.layers = nn.ModuleList()

        for i in range(len(blocks_down)):
            level = nn.ModuleDict()

            kernel_size, padding, stride = aniso_kernel(anisotropic_scales[i]) if anisotropic_scales else (3, 1, 2)
            blocks = [
                SegResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=filters,
                    kernel_size=kernel_size,
                    norm=norm,
                    act=act,
                )
                for _ in range(blocks_down[i])
            ]
            level["blocks"] = nn.Sequential(*blocks)

            if i < len(blocks_down) - 1:
                level["downsample"] = Conv[Conv.CONV, spatial_dims](
                    in_channels=filters,
                    out_channels=2 * filters,
                    bias=False,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            else:
                level["downsample"] = nn.Identity()

            self.layers.append(level)
            filters *= 2

        self.head_module = head_module
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.init_filters = init_filters
        self.norm = norm
        self.act = act
        self.spatial_dims = spatial_dims

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        x = self._standardize(x)
        x = self.conv_init(x)
        for level in self.layers:
            x = level["blocks"](x)
            outputs.append(x)
            x = level["downsample"](x)

        if self.head_module is not None:
            outputs = self.head_module(outputs)

        return outputs

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self._forward(x)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardize the input tensor to have zero mean and unit variance.
        """
        # x = (x - x.mean()) / x.std().clamp(min=1e-5)
        return x - 0.5
        # return x


class SegResNetDS2(nn.Module):
    """
    SegResNetDS based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    It is similar to https://docs.monai.io/en/stable/networks.html#segresnet, with several
    improvements including deep supervision and non-isotropic kernel support.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``BATCH``.
        blocks_down: number of downsample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of upsample blocks (optional).
        dsdepth: number of levels for deep supervision. This will be the length of the list of outputs at each scale level.
                 At dsdepth==1,only a single output is returned.
        preprocess: optional callable function to apply before the model's forward pass
        resolution: optional input image resolution. When provided, the network will first use non-isotropic kernels to bring
                    image spacing into an approximately isotropic space.
                    Otherwise, by default, the kernel size and downsampling is always isotropic.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        act: tuple | str = "relu",
        norm: tuple | str = "instance",
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple | None = None,
        dsdepth: int = 1,
        preprocess: nn.Module | Callable | None = None,
        upsample_mode: UpsampleMode | str = "nontrainable",
        resolution: tuple | None = None,
    ):
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("`spatial_dims` can only be 1, 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.norm = norm
        self.blocks_down = blocks_down
        self.dsdepth = max(dsdepth, 1)
        self.resolution = resolution
        self.preprocess = preprocess

        if resolution is not None:
            if not isinstance(resolution, (list, tuple)):
                raise TypeError("resolution must be a tuple")
            elif not all(r > 0 for r in resolution):
                raise ValueError("resolution must be positive")

        # ensure normalization had affine trainable parameters (if not specified)
        norm = split_args(norm)
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)  # type: ignore

        # ensure activation is inplace (if not specified)
        act = split_args(act)
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)  # type: ignore

        anisotropic_scales = None
        if resolution:
            anisotropic_scales = scales_for_resolution(resolution, n_stages=len(blocks_down))
        self.anisotropic_scales = anisotropic_scales

        self.encoder = SegResEncoder(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
            anisotropic_scales=anisotropic_scales,
        )

    def shape_factor(self):
        """
        Calculate the factors (divisors) that the input image shape must be divisible by
        """
        if self.anisotropic_scales is None:
            d = [2 ** (len(self.blocks_down) - 1)] * self.spatial_dims
        else:
            d = list(np.prod(np.array(self.anisotropic_scales[:-1]), axis=0))
        return d

    def is_valid_shape(self, x):
        """
        Calculate if the input shape is divisible by the minimum factors for the current network configuration
        """
        a = [i % j == 0 for i, j in zip(x.shape[2:], self.shape_factor())]
        return all(a)

    def _forward(self, x: torch.Tensor, with_point, with_auto) -> Union[None, torch.Tensor, list[torch.Tensor]]:
        if self.preprocess is not None:
            x = self.preprocess(x)

        if not self.is_valid_shape(x):
            raise ValueError(f"Input spatial dims {x.shape} must be divisible by {self.shape_factor()}")

        x_down = self.encoder(x)

        x_down.reverse()
        x = x_down[0].clone()

        return x_down

    def forward(
        self, x: torch.Tensor, with_point=False, with_auto=True, **kwargs
    ) -> Union[None, torch.Tensor, list[torch.Tensor]]:
        return self._forward(x, with_point, with_auto)
