from torch import nn

from src.model.modeling.ahus_model import AhusModel
from src.model.modeling.normalized_mask_decoder3D import NormalizedMaskDecoder3D
from src.model.modeling.position_encoder3D import PositionEncoder3D
from src.model.modeling.prompt_encoder3D import PromptEncoder3D
from src.model.modeling.segresnet import SegResNetDS2


def build_ahus_model(position_encoder_class: type[PositionEncoder3D], large: bool = False) -> AhusModel:
    # init_filters = 16 if not large else 8
    init_filters = 8
    blocks_down: tuple = (1, 1, 2, 4)  # if not large else (1, 1, 2, 4, 4)
    blocks_up: tuple = (1, 1, 1, 1)  # if not large else (1, 1, 1, 1, 1)
    # embed_dim = 128
    embed_dim = 64
    num_heads = 4
    segresnet = SegResNetDS2(init_filters=init_filters, blocks_down=blocks_down)

    position_encoder = position_encoder_class(embed_dim=embed_dim, num_heads=num_heads)

    prompt_encoder = PromptEncoder3D(
        embed_dim=embed_dim,
        prev_mask_downscaling_factor=8 if not large else 16,
        init_filters=init_filters,
        position_encoder=position_encoder,
    )
    mask_decoder = NormalizedMaskDecoder3D(
        embed_dim=embed_dim,
        depth=2,
        num_heads=num_heads,
        out_chans=1,
        activation=nn.SiLU,
        init_filters=init_filters,
        blocks_up=blocks_up,
        upsample_mode="nontrainable",
    )
    model = AhusModel(
        segresnet=segresnet,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )
    return model


# if __name__ == "__main__":
#     model = model_registry["ahus_model_sinusoidal"]()
#     print(model)
