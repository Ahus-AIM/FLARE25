from torch import nn

from src.model.modeling.normalized_mask_decoder3D import NormalizedMaskDecoder3D
from src.model.modeling.prompt_encoder3D import PromptEncoder3D
from src.model.modeling.segresnet import SegResNetDS2


class AhusModel(nn.Module):
    def __init__(
        self, segresnet: SegResNetDS2, prompt_encoder: PromptEncoder3D, mask_decoder: NormalizedMaskDecoder3D
    ) -> None:
        super().__init__()
        self.segresnet = segresnet
        self.image_encoder = segresnet.encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
