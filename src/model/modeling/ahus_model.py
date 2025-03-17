from torch import nn


class AhusModel(nn.Module):
    def __init__(self, segresnet: nn.Module, prompt_encoder: nn.Module, mask_decoder: nn.Module) -> None:
        super().__init__()
        self.segresnet = segresnet
        self.image_encoder = segresnet.encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
