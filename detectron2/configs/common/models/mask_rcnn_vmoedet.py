from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.modeling import SimpleFeaturePyramid, VMoE
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .mask_rcnn_fpn import model
from ..data.constants import constants

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

# Creates Simple Feature Pyramid from VMoE-B backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(VMoE)(img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), is_student=False, moe_pos=[0,1,2,3,4,5,6,7,8,9,10,11], 
        divided_depths=[3,3,3,3], out_feature="last_feat"),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=224,
)

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]
