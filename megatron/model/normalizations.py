from .rms_norm import RMSNorm
from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm


NORMALIZATIONS = {
    "layernorm": LayerNorm,
    "rmsnorm": RMSNorm,
}
