from functools import partial

from torch import nn

from .modeling import AhusModel, NormalizedMaskDecoder3D, PromptEncoder3D, SegResNetDS2
from .modeling.position_encoder3D import LieRE, PositionEmbeddingRandom3D, PositionEncoder3D, RoPEMixed


def build_ahus_model(position_encoder_class: PositionEncoder3D) -> AhusModel:
    init_filters = 16
    blocks_down: tuple = (1, 1, 1, 1)  # should be 1, 2, 2, 4
    blocks_up: tuple = (1, 1, 1, 1)
    embed_dim = 128
    num_heads = 8
    segresnet = SegResNetDS2(init_filters=init_filters, blocks_down=blocks_down)

    if position_encoder_class == RoPEMixed:
        position_encoder = RoPEMixed(embed_dim=embed_dim, num_heads=num_heads)
    elif position_encoder_class == LieRE:
        position_encoder = LieRE(embed_dim=embed_dim, num_heads=num_heads)
    else:
        position_encoder = PositionEmbeddingRandom3D(embed_dim // 2)

    prompt_encoder = PromptEncoder3D(
        embed_dim=embed_dim,
        prev_mask_downscaling_factor=8,
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
    )
    model = AhusModel(
        segresnet=segresnet,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )
    return model


model_registry = {
    "ahus_model_sinusoidal": partial(build_ahus_model, PositionEmbeddingRandom3D),
    "ahus_model_rope_mixed": partial(build_ahus_model, RoPEMixed),
    "ahus_model_liere": partial(build_ahus_model, LieRE),
}

if __name__ == "__main__":
    model = model_registry["ahus_model_sinusoidal"]()
    print(model)
