import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial

from fmoe.transformer import FMoETransformerMLP
from fmoe.gates import NaiveGate

from deep_incubation.timm.models.registry import register_model
from deep_incubation.timm.models.layers import DropPath, trunc_normal_
from deep_incubation.timm.models.vision_transformer import Mlp, Attention, PatchEmbed, HybridEmbed
from deep_incubation.timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = [
    'vmoe_student_base', 'vmoe_base', 
    'vmoe_student_large', 'vmoe_large',
    'vmoe_student_huge', 'vmoe_huge',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }
    

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 pre_stitch=False, post_stitch=False, stitch_dim=None):
        super().__init__()
        self.pre_stitch = None
        if pre_stitch:
            self.pre_stitch =  nn.Sequential(OrderedDict([
                ('ln', nn.LayerNorm(stitch_dim)),
                ('fc', nn.Linear(stitch_dim, dim)),])
            )
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.post_stitch = None
        if post_stitch:
            self.post_stitch =  nn.Sequential(OrderedDict([
                ('ln', nn.LayerNorm(dim)),
                ('fc', nn.Linear(dim, stitch_dim)),])
            )

    def forward(self, x):
        if self.pre_stitch:
            x = self.pre_stitch(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.post_stitch:
            x = self.post_stitch(x)
        return x
    

class MoeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_expert=32, world_size=1,
                 gate=NaiveGate, set_gate_hook=False, pre_stitch=False, post_stitch=False, stitch_dim=None):
        super().__init__()
        self.pre_stitch = None
        if pre_stitch:
            self.pre_stitch =  nn.Sequential(OrderedDict([
                ('ln', nn.LayerNorm(stitch_dim)),
                ('fc', nn.Linear(stitch_dim, dim)),])
            )
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        moe_hidden_dim = int(dim * mlp_ratio)
        self.moe = FMoETransformerMLP(num_expert=num_expert, world_size=world_size, d_model=dim, d_hidden=moe_hidden_dim, 
                                      activation=act_layer(), gate=gate, gate_hook=self.gate_hook_fn if set_gate_hook else None,)
        self.post_stitch = None
        if post_stitch:
            self.post_stitch =  nn.Sequential(OrderedDict([
                ('ln', nn.LayerNorm(dim)),
                ('fc', nn.Linear(dim, stitch_dim)),])
            )

    def forward(self, x):
        if self.pre_stitch:
            x = self.pre_stitch(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.moe(self.norm2(x)))
        if self.post_stitch:
            x = self.post_stitch(x)
        return x
    
    def gate_hook_fn(self, gate_top_k_idx, gate_score, kwargs=None):
        pass


class VMoE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, divided_depths=None, 
                 is_student=False, idx=None, moe_pos=None, num_expert=8, gate=NaiveGate, set_gate_hook=False, 
                 pre_stitch_pos=None, post_stitch_pos=None, stitch_dim=None):
        super().__init__()

        assert divided_depths is not None, 'Please specify depth for each module'
        depth = sum(divided_depths)
        self.div = divided_depths
        self.stages = nn.ModuleList([nn.Module() for _ in range(len(divided_depths))])
        self.is_student = is_student
        self.idx = idx
        self.moe_pos = moe_pos
        self.num_expert = num_expert
        self.set_gate_hook = set_gate_hook

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if not is_student or self.idx == 0:
            if hybrid_backbone is not None:
                self.patch_embed = HybridEmbed(
                    hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            pre_stitch = False
            post_stitch = False
            if pre_stitch_pos and i in pre_stitch_pos:
                pre_stitch = True
            if post_stitch_pos and i in post_stitch_pos:
                post_stitch = True

            if moe_pos and i in moe_pos:
                self.blocks.append(
                    MoeBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                        num_expert=num_expert, gate=gate, set_gate_hook=set_gate_hook,
                        pre_stitch=pre_stitch, post_stitch=post_stitch, stitch_dim=stitch_dim
                    )
                )
            else:
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        pre_stitch=pre_stitch, post_stitch=post_stitch, stitch_dim=stitch_dim
                        )
                    )
        
        if not is_student or self.idx == -1:
            self.norm = norm_layer(embed_dim)

            # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
            #self.repr = nn.Linear(embed_dim, representation_size)
            #self.repr_act = nn.Tanh()

            # Classifier head
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # register modules for module incubation & model assembling
        self._register_modules()

    def _register_modules(self):
        divided_modules_iter = iter(zip(self.div, self.stages))
        blocks_to_add, div_module = next(divided_modules_iter)

        if not self.is_student or self.idx == 0:
            div_module.pos_embed = self.pos_embed
            div_module.cls_token = self.cls_token
            div_module.patch_embed = self.patch_embed
        for block in self.blocks:
            if hasattr(div_module, 'blocks'):
                div_module.blocks.append(block)
            else:
                div_module.blocks = nn.ModuleList([block])
            blocks_to_add -= 1
            if blocks_to_add == 0:
                try:
                    blocks_to_add, div_module = next(divided_modules_iter)
                except StopIteration:
                    pass
        if not self.is_student or self.idx == -1:
            div_module.norm = self.norm
            div_module.head = self.head

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_gate_loss(self):
        gate_loss = torch.tensor(0.).cuda()
        for blk in self.blocks:
            if not isinstance(blk, MoeBlock):
                continue
            gate_loss += blk.moe.gate.get_loss()
        return gate_loss

    def forward_features(self, x):
        if not self.is_student or self.idx == 0:
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if not self.is_student or self.idx == -1:
            x = self.norm(x)
            return x[:, 0]
        
        return x
        
    def forward(self, x):
        x = self.forward_features(x)
        if not self.is_student or self.idx == -1:
            x = self.head(x)
        return x


@register_model
def vmoe_student_base(pretrained=False, **kwargs):
    model = VMoE(
        patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), is_student=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vmoe_base(pretrained=False, **kwargs):
    model = VMoE(
        patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), is_student=False, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vmoe_student_large(pretrained=False, **kwargs):
    model = VMoE(
        patch_size=16, embed_dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), is_student=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vmoe_large(pretrained=False, **kwargs):
    model = VMoE(
        patch_size=16, embed_dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), is_student=False, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vmoe_student_huge(pretrained=False, **kwargs):
    model = VMoE(
        patch_size=14, embed_dim=1280, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), is_student=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vmoe_huge(pretrained=False, **kwargs):
    model = VMoE(
        patch_size=14, embed_dim=1280, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), is_student=False, **kwargs)
    model.default_cfg = _cfg()
    return model