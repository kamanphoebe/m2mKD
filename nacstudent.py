import torch
import torch.nn as nn
import einops as eo
import math
from typing import Optional, List

from deep_incubation.timm.models.registry import register_model

from nacs.neural_compilers import utils
import nacs.neural_compilers.model.nc.tokenizers as tkn
from nacs.neural_compilers.model.nc.attention import ModFFN, ReadInAttention
from nacs.neural_compilers.model.nc.utils import LayerScale, DropPath
from nacs.neural_compilers.utils import override, keep_valid_backups, StatelessWrapper, ModSequential


__all__ = ['mediator', 'readin', 'readout']


class ModulatedLatentAttention(nn.Module):
    """
    KernelModulatedLatentAttention() of NACs without kernel computation.
    """
    def __init__(
            self,
            *,
            state_dim: int,
            code_dim: int,
            num_heads: int = 1,
            head_dim: int = 64,
            include_residual: bool = True,
            path_drop_prob: float = 0.0,
            layer_scale_initial_value: float = 1e-6,
            scale_attention_by_affinities_after_softmax: bool = False,
            share_layernorm: bool = False,
            qkv_bias: bool = True,
            mod_fc_cls: str = "ModFC",
            mod_fc_kwargs: Optional[dict] = None):
        
        super(ModulatedLatentAttention, self).__init__()
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.include_residual = include_residual
        self.scale_attention_by_affinities_after_softmax = (
            scale_attention_by_affinities_after_softmax
        )
        self.share_layernorm = share_layernorm
        self.qkv_bias = qkv_bias
        # Define the modules
        self.receiver_layernorm = nn.LayerNorm(self.state_dim)
        if share_layernorm:
            self.sender_layernorm = None
        else:
            self.sender_layernorm = nn.LayerNorm(self.state_dim)
        mod_fc_cls = getattr(utils, mod_fc_cls)
        self.query_compiler = mod_fc_cls(
            in_features=self.state_dim,
            out_features=self.inner_dim,
            context_features=self.code_dim,
            bias=self.qkv_bias,
            **(mod_fc_kwargs or {}),
        )
        self.key_compiler = mod_fc_cls(
            in_features=self.state_dim,
            out_features=self.inner_dim,
            context_features=self.code_dim,
            bias=self.qkv_bias,
            **(mod_fc_kwargs or {}),
        )
        self.value_compiler = mod_fc_cls(
            in_features=self.state_dim,
            out_features=self.inner_dim,
            context_features=self.code_dim,
            bias=self.qkv_bias,
            **(mod_fc_kwargs or {}),
        )
        self.exit_layer = mod_fc_cls(
            in_features=self.inner_dim,
            out_features=self.state_dim,
            context_features=self.code_dim,
            **(mod_fc_kwargs or {}),
        )
        self.drop_path = DropPath(drop_prob=path_drop_prob)
        self.layer_scale = LayerScale(
            self.state_dim, initial_value=layer_scale_initial_value
        )

    @property
    def inner_dim(self):
        return self.head_dim * self.num_heads

    def forward(
        self,
        # A query "receives", whereas the key/value "sends" a message.
        receiver_states: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_codes: torch.Tensor,
    ):
        # Shape formats are the following
        # (B: batch; {U,V}: set elements; C: channels; H: heads)
        #   receiver_states: BUC
        #   receiver_codes: UC
        #   sender_states: BVC or BVC
        #   sender_codes: VC
        # --------------------
        # Pre-layernorm
        receiver_states_post_layernorm = self.receiver_layernorm(receiver_states)
        if self.sender_layernorm is not None:
            sender_states_post_layernorm = self.sender_layernorm(sender_states)
        else:
            assert self.share_layernorm
            sender_states_post_layernorm = self.receiver_layernorm(sender_states)
        # --------------------
        # Compute queries, keys and values
        # keys.shape = values.shape = BV(HC)
        keys = self.key_compiler(sender_states_post_layernorm, sender_codes)
        values = self.value_compiler(sender_states_post_layernorm, sender_codes)
        # queries.shape = BU(HC)
        queries = self.query_compiler(receiver_states_post_layernorm, receiver_codes)
        # Unfold the head and channel axes
        # shapes = BH{U,V}C
        queries, keys, values = map(
            lambda x: eo.rearrange(x, "b u (h c) -> b h u c", h=self.num_heads),
            [queries, keys, values],
        )
        # --------------------
        # Compute scaled dot product attention
        scale = math.sqrt(self.head_dim)
        attention_scores = torch.einsum("bhuc,bhvc->bhuv", queries, keys) / scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # --------------------
        # Compute the aggregated messages from values
        aggregated_messages = torch.einsum("bhuv,bhvc->bhuc", attention_weights, values)
        # Unfold heads
        aggregated_messages = eo.rearrange(aggregated_messages, "b h u c -> b u (h c)")
        # --------------------
        # Run through the final layer
        aggregated_messages = self.exit_layer(aggregated_messages, receiver_codes)
        # --------------------
        # Apply LayerScale
        aggregated_messages = self.layer_scale(aggregated_messages)
        # --------------------
        # Apply the residual
        if self.include_residual:
            output = self.drop_path(old=receiver_states, residual=aggregated_messages)
        else:
            assert (
                self.drop_path.drop_prob == 0.0
            ), "Cannot apply drop-path if not applying residuals."
            output = aggregated_messages
        # --------------------
        # Done!
        return output


class AttentiveReadIn(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        code_dim: int,
        num_heads: int = 1,
        head_dim: int = 64,
        path_drop_prob: float = 0.0,
        read_in_layer_scale_initial_value: float = 1.0,
        ffn_layer_scale_initial_value: float = 1e-6,
        ffn_capacity_factor: float = 1.0,
        include_residual_in_read_in_attention: bool = False,
        use_geglu: bool = False,
        qkv_bias_in_attention: bool = True,
        read_in_attention_kwargs: Optional[dict] = None,
        ffn_kwargs: Optional[dict] = None,
        mod_fc_cls: str = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
        ffn_mod_fc_cls: Optional[str] = None,
        ffn_mod_fc_kwargs: Optional[dict] = None,
    ):
        super(AttentiveReadIn, self).__init__()
        self.read_in_attention = ReadInAttention(
            **override(
                read_in_attention_kwargs,
                input_dim=input_dim,
                state_dim=state_dim,
                code_dim=code_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                include_residual=include_residual_in_read_in_attention,
                layer_scale_initial_value=read_in_layer_scale_initial_value,
                qkv_bias=qkv_bias_in_attention,
                mod_fc_cls=mod_fc_cls,
                mod_fc_kwargs=mod_fc_kwargs,
            )
        )
        if ffn_mod_fc_cls is not None:
            # Don't wanna use the kwargs for the wrong class
            ffn_mod_fc_kwargs = ffn_mod_fc_kwargs or {}
        else:
            ffn_mod_fc_cls, ffn_mod_fc_kwargs = mod_fc_cls, mod_fc_kwargs
        self.ffn = ModFFN(
            **override(
                ffn_kwargs,
                state_dim=state_dim,
                code_dim=code_dim,
                capacity_factor=ffn_capacity_factor,
                include_residual=True,
                path_drop_prob=path_drop_prob,
                layer_scale_initial_value=ffn_layer_scale_initial_value,
                use_geglu=use_geglu,
                mod_fc_cls=ffn_mod_fc_cls,
                mod_fc_kwargs=ffn_mod_fc_kwargs,
            ))

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_codes: Optional[torch.Tensor] = None,
    ):
        receiver_states = self.read_in_attention(
            receiver_states=receiver_states, 
            receiver_signatures=None,
            receiver_codes=receiver_codes, 
            sender_states=sender_states
        )

        receiver_states = self.ffn(receiver_states, receiver_codes)

        return receiver_states
    

class MediatorStudent(nn.Module):
    """
    Propagator() of NACs without connection between modules.
    """
    def __init__(
            self,
            stitch_dim: int,
            state_dim: int,
            code_dim: int,
            num_heads: int = 6,
            head_dim: int = 64,
            path_drop_prob: float = 0.0,
            layer_scale_initial_value: float = 1e-6,
            ffn_capacity_factor: List[float] = [],
            latent_attention_kwargs: Optional[dict] = None,
            include_residual_in_latent_attention: bool = True,
            use_geglu: bool = False,
            qkv_bias_in_attention: bool = True,
            ffn_kwargs: Optional[dict] = None,
            mod_fc_cls: Optional[str] = "ModFC",
            mod_fc_kwargs: Optional[dict] = None,
            ffn_mod_fc_cls: Optional[str] = None,
            **share_kwargs,
        ):

        super(MediatorStudent, self).__init__()
        
        self.pre_stitch = nn.Linear(stitch_dim, state_dim)

        self.latent_attention = ModulatedLatentAttention(
            **override(
                latent_attention_kwargs,
                state_dim=state_dim,
                code_dim=code_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                include_residual=include_residual_in_latent_attention,
                layer_scale_initial_value=layer_scale_initial_value,
                path_drop_prob=path_drop_prob,
                qkv_bias=qkv_bias_in_attention,
                mod_fc_cls=mod_fc_cls,
                mod_fc_kwargs=mod_fc_kwargs,
            )
        )
        if ffn_mod_fc_cls is not None:
            # Don't wanna use the kwargs for the wrong class
            ffn_mod_fc_kwargs = ffn_mod_fc_kwargs or {}
        else:
            ffn_mod_fc_cls, ffn_mod_fc_kwargs = mod_fc_cls, mod_fc_kwargs
        self.ffn = ModFFN(
            **override(
                ffn_kwargs,
                state_dim=state_dim,
                code_dim=code_dim,
                capacity_factor=ffn_capacity_factor,
                include_residual=True,
                path_drop_prob=path_drop_prob,
                layer_scale_initial_value=layer_scale_initial_value,
                use_geglu=use_geglu,
                mod_fc_cls=ffn_mod_fc_cls,
                mod_fc_kwargs=ffn_mod_fc_kwargs,
            )
        )
        
        self.post_stitch = nn.Linear(state_dim, stitch_dim)

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_codes: Optional[torch.Tensor] = None,
    ):  
        receiver_states = self.pre_stitch(receiver_states)
        sender_states = self.pre_stitch(sender_states)
        receiver_states = self.latent_attention(
            receiver_states,
            receiver_codes,
            sender_states,
            sender_codes,
        )
        receiver_states = self.ffn(receiver_states, receiver_codes)
        receiver_states = self.post_stitch(receiver_states)
        return receiver_states
    

class ReadInStudent(nn.Module):
    def __init__(
            self,
            input_dim: int,
            stitch_dim: int,
            num_heads: int = 6,
            head_dim: int = 64, 
            tokenizer_type: str = "ConvNeXtImageTokenizer",
            tokenizer_kwargs: Optional[dict] = None,
            read_in_cls: str = "AttentiveReadIn",
            **share_kwargs,
        ):

        super(ReadInStudent, self).__init__()

        # Initialize tokenizer
        tokenizer_cls = getattr(tkn, tokenizer_type)
        tokenizer_kwargs = keep_valid_backups(
            tokenizer_type,
            tokenizer_kwargs,
            input_dim=input_dim,
            repr_dim=share_kwargs['state_dim'],
        )
        self.tokenizer = tokenizer_cls(**tokenizer_kwargs)

        # Initialize ReadIn
        read_in_dim = self.tokenizer.output_dim
        read_in_cls = {
            "AttentiveReadIn": AttentiveReadIn,
            # We only support AttentiveReadIn so far.
            # "ProjectionReadIn": ProjectionReadIn,
        }[read_in_cls]
        readin_kwargs = keep_valid_backups(
            read_in_cls,
            {'input_dim': read_in_dim,
             'num_heads': num_heads,
             'head_dim': head_dim},
            **share_kwargs,
        )
        self.read_in = read_in_cls(**readin_kwargs)
        self.post_stitch = nn.Linear(share_kwargs["state_dim"], stitch_dim)
    
    def forward(
            self, 
            x,
            receiver_states: torch.Tensor,
            receiver_codes: torch.Tensor,
        ):
        output = self.tokenizer(x)
        output = self.read_in(receiver_states, receiver_codes, output)
        output = self.post_stitch(output)
        return output
    

class ReadOutStudent(MediatorStudent):
    def __init__(
        self,
        num_heads: List[int],
        head_dim: List[int], 
        output_dim: Optional[int] = None,
        pre_output_layernorm: bool = True,
        use_head: bool = True,
        include_residual_in_latent_attention: bool = False,
        tokenizer_type: str = "FirstSlotAsLogits",
        tokenizer_kwargs: Optional[dict] = None,
        **share_kwargs,
    ):
        super(ReadOutStudent, self).__init__(
            **share_kwargs, 
            num_heads=num_heads,
            head_dim=head_dim,
            include_residual_in_latent_attention=include_residual_in_latent_attention,
        )
        self.state_dim = share_kwargs["state_dim"]
        self.code_dim = share_kwargs["code_dim"]
        self.output_dim = output_dim
        self.pre_output_layernorm = pre_output_layernorm
        self.use_head = use_head
        del self.post_stitch
        # Initialize ReadOut
        mod_fc_cls = getattr(utils, share_kwargs.get("mod_fc_cls", "ModFC"))
        mod_fc_kwargs = share_kwargs.get("mod_fc_kwargs", None)
        if self.use_head:
            assert self.output_dim is not None
            self.head = ModSequential(
                StatelessWrapper(
                    (
                        nn.LayerNorm(self.state_dim)
                        if pre_output_layernorm
                        else nn.Identity()
                    )
                ),
                mod_fc_cls(
                    self.state_dim,
                    self.output_dim,
                    self.code_dim,
                    **(mod_fc_kwargs or {}),
                ),
            )
        else:
            # output_dim is None, meaning we have a no-op
            self.head = StatelessWrapper(nn.Identity())
        # Initialize tokenizer
        tokenizer_cls = getattr(tkn, tokenizer_type)
        #   kwargs
        tokenizer_kwargs = keep_valid_backups(
            tokenizer_cls,
            tokenizer_kwargs,
            state_dim=self.state_dim,
            output_dim=output_dim,
        )
        self.tokenizer = tokenizer_cls(**tokenizer_kwargs)

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_codes: Optional[torch.Tensor] = None,
    ):
        sender_states = self.pre_stitch(sender_states)
        receiver_states = self.latent_attention(
            receiver_states,
            receiver_codes,
            sender_states,
            sender_codes,
        )
        receiver_states = self.ffn(receiver_states, receiver_codes)
        receiver_states = self.head(receiver_states, receiver_codes)
        output = self.tokenizer(receiver_states)
        return output
    

@register_model
def mediator(**kwargs):
    model = MediatorStudent(**kwargs)
    return model


@register_model
def readin(**kwargs):
    model = ReadInStudent(**kwargs)
    return model


@register_model
def readout(**kwargs):
    model = ReadOutStudent(**kwargs)
    return model