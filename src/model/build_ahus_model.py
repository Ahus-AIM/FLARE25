from torch import nn

from .modeling import (
    AhusModel,
    NormalizedMaskDecoder3D,
    PromptEncoder3D,
    SegResNetDS2,
)


def build_ahus_model():
    init_filters = 16
    blocks_down: tuple = (1, 1, 1, 1)  # should be 1, 2, 2, 4
    blocks_up: tuple = (1, 1, 1, 1)
    embed_dim = 128
    segresnet = SegResNetDS2(init_filters=init_filters, blocks_down=blocks_down)
    prompt_encoder = PromptEncoder3D(
        embed_dim=embed_dim,
        prev_mask_downscaling_factor=8,
        image_embedding_size=(16, 16, 16),
        input_image_size=(128, 128, 128),
        activation=nn.GELU,
        init_filters=init_filters,
    )
    mask_decoder = NormalizedMaskDecoder3D(
        embed_dim=embed_dim,
        depth=2,
        num_heads=8,
        out_chans=1,
        input_size=16,
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


model_registry = {"ahus_model": build_ahus_model}

if __name__ == "__main__":
    model = model_registry["ahus_model"]()
    print(model)
