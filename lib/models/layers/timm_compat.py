try:
    from timm.layers import DropPath, Mlp, lecun_normal_, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, Mlp, lecun_normal_, to_2tuple, trunc_normal_

try:
    from timm.models import adapt_input_conv, build_model_with_cfg, named_apply, register_model
except ImportError:
    from timm.models.helpers import adapt_input_conv, build_model_with_cfg, named_apply
    from timm.models.registry import register_model
