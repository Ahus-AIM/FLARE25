from functools import partial

from src.model.build_ahus_model import build_ahus_model
from src.model.modeling.position_encoder3D import LieRE, PositionEmbeddingRandom3D, RoPEMixed

model_registry = {
    "ahus_model_sinusoidal": partial(build_ahus_model, PositionEmbeddingRandom3D),
    "ahus_model_rope_mixed": partial(build_ahus_model, RoPEMixed),
    "ahus_model_rope_mixed_large": partial(build_ahus_model, RoPEMixed, large=True),
    "ahus_model_liere": partial(build_ahus_model, LieRE),
    "ahus_model_liere_large": partial(build_ahus_model, LieRE, large=True),
}
