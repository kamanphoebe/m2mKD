import math
from typing import Optional, List, OrderedDict

import torch
import torch.nn as nn
import einops as eo
#from torch_db.tracer import Tracer

from neural_compilers.model.nc.kernels import get_kernel
from neural_compilers.model.nc.utils import DropPath, LayerScale, VectorScaleNoise
from neural_compilers.utils import ModSequential, StatelessWrapper, override, GEGLU
from neural_compilers import utils

"""
@Tracer.register_tracer_probes(
    "pre_mask_attention_scores",
    "post_mask_attention_scores",
    "post_mask_attention_weights",
)"""
class KernelModulatedLatentAttention(nn.Module):
    """
    Kernel Modulated Latent Attention generalizes attentive message passing
    from all-to-all graphs to a learned graph. As with message passing,
    there is a sender and a receiver. A sender sends a message only to
    receivers whose signatures lie in vicinity to that of the sender. In
    the language of vanilla self attention, the senders generate the keys
    and the values, whereas the receivers generate the queries.

    Further, the identity of the sender and the receiver is infused in
    via their respective codes.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        code_dim: int,
        num_heads: int = 1,
        head_dim: int = 64,
        include_residual: bool = True,
        kernel_type: str = "DotProductKernel",
        kernel_kwargs: Optional[dict] = None,
        mask_attn_scores_with_affinities: bool = True,
        path_drop_prob: float = 0.0,
        layer_scale_initial_value: float = 1e-6,
        scale_attention_by_affinities_after_softmax: bool = False,
        normalize_kernel_along_keys: bool = True,
        share_layernorm: bool = False,
        qkv_bias: bool = True,
        mod_fc_cls: str = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
    ):
        super(KernelModulatedLatentAttention, self).__init__()
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.include_residual = include_residual
        self.mask_attn_scores_with_affinities = mask_attn_scores_with_affinities
        self.scale_attention_by_affinities_after_softmax = (
            scale_attention_by_affinities_after_softmax
        )
        self.normalize_kernel_along_keys = normalize_kernel_along_keys
        self.share_layernorm = share_layernorm
        self.qkv_bias = qkv_bias
        # Define the modules
        self.kernel = get_kernel(kernel_type)(**(kernel_kwargs or {}))
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
        receiver_signatures: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_signatures: torch.Tensor,
        sender_codes: torch.Tensor,
    ):
        # Shape formats are the following
        # (B: batch; {U,V}: set elements; C: channels; H: heads)
        #   receiver_states: BUC
        #   receiver_signatures: UC or BUC
        #   receiver_codes: UC
        #   sender_states: BVC or BVC
        #   sender_signatures: VC
        #   sender_codes: VC
        # --------------------
        batch_size = sender_states.shape[0]
        # Compute the kernel
        # log_affinities.shape = affinities.shape = UV or BUV
        if self.mask_attn_scores_with_affinities:
            with self.kernel.return_log_kernel():
                log_affinities = self.kernel(
                    receiver_signatures,
                    sender_signatures,
                    normalize_kernel_along_dim=(
                        -1 if self.normalize_kernel_along_keys else None
                    ),
                    batch_size=batch_size,
                )
            affinities = log_affinities.exp()
        else:
            # This code-path supports additional features like sampling
            affinities = self.kernel(
                receiver_signatures,
                sender_signatures,
                normalize_kernel_along_dim=(
                    -1 if self.normalize_kernel_along_keys else None
                ),
                batch_size=batch_size,
            )
            log_affinities = None
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
        # Track for analysis
        """attention_scores = Tracer.track(
            self, "pre_mask_attention_scores", attention_scores
        )"""
        if self.mask_attn_scores_with_affinities:
            assert log_affinities is not None
            # attention_scores.shape = BHUV
            # log_affinities.shape = UV or BUV
            if log_affinities.dim() == 3:
                # log_affinities.shape = BUV
                # Reshape to B1UV
                log_affinities = log_affinities[:, None, :, :]
            elif log_affinities.dim() == 2:
                # log_affinities.shape = UV
                # Reshape to 11UV
                log_affinities = log_affinities[None, None, :, :]
            else:
                raise TypeError(f"Invalid log affinity shape: {log_affinities.shape}")
            # log_affinities.shape = 11UV or B1UV
            assert log_affinities.dim() == 4
            attention_scores = attention_scores + log_affinities
        # Track for analysis
        """attention_scores = Tracer.track(
            self, "post_mask_attention_scores", attention_scores
        )"""
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # --------------------
        # Modulate the attention scores. Note that unlike masking, modulation scales
        # the values.
        if self.scale_attention_by_affinities_after_softmax:
            # affinities.shape = UV or BUV
            einsum_program = {
                2: "bhuv,uv->bhuv",
                3: "bhuv,buv->bhuv",
            }[affinities.dim()]
            attention_weights = torch.einsum(
                einsum_program, attention_weights, affinities
            )
        # Track for analysis
        """attention_weights = Tracer.track(
            self, "post_mask_attention_weights", attention_weights
        )"""
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


class ReadInAttention(nn.Module):
    """
    This might look like a typical Perceiver-style cross attention, but it isn't.

    In the typical cross attention, the query latent doesn't influence the
    computation that is performed on the input (i.e. the key and value projection).
    But here, each query latent has a different key and value projector (applied
    on the inputs), where "different" is in the sense that the key/value projectors
    are conditioned by the codes of the query latents.

    Is this expensive? Yes, very much so. But unlike in Perceivers where all latents
    attend to the inputs, we can only have fewer input latents attending to the inputs.
    Further, we can use a deeper backbone stem to reduce the number of input set
    elements to attend to.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        state_dim: int,
        code_dim: int,
        num_heads: int = 1,
        head_dim: int = 64,
        include_residual: bool = True,
        layer_scale_initial_value: float = 1.0,
        qkv_bias: bool = True,
        mod_fc_cls: str = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
    ):
        super(ReadInAttention, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.include_residual = include_residual
        self.qkv_bias = qkv_bias
        # Define the modules
        self.sender_layernorm = nn.LayerNorm(self.input_dim)
        self.receiver_layernorm = nn.LayerNorm(self.state_dim)
        mod_fc_cls = getattr(utils, mod_fc_cls)
        self.query_compiler = mod_fc_cls(
            in_features=self.state_dim,
            out_features=self.inner_dim,
            context_features=self.code_dim,
            bias=self.qkv_bias,
            **(mod_fc_kwargs or {}),
        )
        self.key_compiler = mod_fc_cls(
            in_features=self.input_dim,
            out_features=self.inner_dim,
            context_features=self.code_dim,
            bias=self.qkv_bias,
            **(mod_fc_kwargs or {}),
        )
        self.value_compiler = mod_fc_cls(
            in_features=self.input_dim,
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
        self.layer_scale = LayerScale(
            self.state_dim, initial_value=layer_scale_initial_value
        )

    @property
    def inner_dim(self):
        return self.head_dim * self.num_heads

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_signatures: Optional[torch.Tensor],
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_signatures: Optional[torch.Tensor] = None,
        sender_codes: Optional[torch.Tensor] = None,
    ):
        # Shape formats are the following
        # (B: batch; {U,V}: set elements; C: channels; H: heads)
        #   receiver_states: BUC
        #   receiver_signatures: UC
        #   receiver_codes: UC
        #   sender_states: BVC
        _, u, _ = receiver_states.shape
        # --------------------
        # Pre-layernorm
        receiver_states_post_layernorm = self.receiver_layernorm(receiver_states)
        sender_states_post_layernorm = self.sender_layernorm(sender_states)
        # --------------------
        # Compute the keys, values and queries
        # This is different from what's usually found in vanilla perceivers.
        # In perceivers, the computation applied to the inputs (to obtain
        # keys and queries) is shared amongst all latents. This is different
        # here: each latent defines its own key and value projectors via their
        # respective codes.
        sender_states_post_layernorm = eo.repeat(
            sender_states_post_layernorm, "b v c -> b v u c", u=u
        )
        # keys.shape = values.shape = BVU(HC)
        keys = self.key_compiler(sender_states_post_layernorm, receiver_codes)
        values = self.value_compiler(sender_states_post_layernorm, receiver_codes)
        # Fold out the head axis
        keys, values = map(
            lambda x: eo.rearrange(x, "b v u (h c) -> b h u v c", h=self.num_heads),
            [keys, values],
        )
        # queries.shape = BU(HC) --> BHUC
        queries = self.query_compiler(receiver_states_post_layernorm, receiver_codes)
        queries = eo.rearrange(queries, "b u (h c) -> b h u c", h=self.num_heads)
        # --------------------
        # Compute scaled dot product attention
        scale = math.sqrt(self.head_dim)
        attention_scores = torch.einsum("bhuc,bhuvc->bhuv", queries, keys) / scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # --------------------
        # Compute the aggregated messages from values
        aggregated_messages = torch.einsum(
            "bhuv,bhuvc->bhuc", attention_weights, values
        )
        # Fold in the heads
        aggregated_messages = eo.rearrange(aggregated_messages, "b h u c -> b u (h c)")
        # --------------------
        # Apply the exit layer
        aggregated_messages = self.exit_layer(aggregated_messages, receiver_codes)
        # --------------------
        # Apply layer scale
        aggregated_messages = self.layer_scale(aggregated_messages)
        # --------------------
        # Apply the residual
        if self.include_residual:
            output = receiver_states + aggregated_messages
        else:
            output = aggregated_messages
        # --------------------
        # Done!
        return output


class ModFFN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        code_dim: int,
        capacity_factor: float = 1.0,
        include_residual: bool = True,
        path_drop_prob: float = 0.0,
        layer_scale_initial_value: float = 1e-6,
        use_geglu: bool = False,
        mod_fc_cls: str = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
    ):
        super(ModFFN, self).__init__()
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.capacity_factor = capacity_factor
        self.include_residual = include_residual
        # Modules
        self.layernorm = nn.LayerNorm(self.state_dim)
        mod_fc_cls = getattr(utils, mod_fc_cls)
        if use_geglu:
            channel_multiplier = 2
            activation_fn = GEGLU()
        else:
            channel_multiplier = 1
            activation_fn = nn.GELU()
        self.ffn = ModSequential(
            mod_fc_cls(
                self.state_dim,
                self.hidden_dim * channel_multiplier,
                self.code_dim,
                **(mod_fc_kwargs or {}),
            ),
            StatelessWrapper(activation_fn),
            mod_fc_cls(
                self.hidden_dim, self.state_dim, self.code_dim, **(mod_fc_kwargs or {})
            ),
        )
        self.drop_path = DropPath(drop_prob=path_drop_prob)
        self.layer_scale = LayerScale(
            self.state_dim, initial_value=layer_scale_initial_value
        )

    @property
    def hidden_dim(self):
        return round(self.capacity_factor * self.state_dim)

    def forward(self, states: torch.Tensor, codes: torch.Tensor):
        residual = self.layernorm(states)
        residual = self.ffn(residual, codes)
        residual = self.layer_scale(residual)
        if self.include_residual:
            output = self.drop_path(old=states, residual=residual)
        else:
            output = residual
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
        code_noise_scale: Optional[float] = None,
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
        if code_noise_scale is not None:
            self.code_noiser = VectorScaleNoise(code_noise_scale)
        else:
            self.code_noiser = None

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_signatures: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_signatures: Optional[torch.Tensor] = None,
        sender_codes: Optional[torch.Tensor] = None,
    ):
        # Remember, residuals are included in the modules so we
        # just need to stack 'em up.
        # But first, add some noise to the code if needed
        if self.code_noiser is not None:
            receiver_codes = self.code_noiser(receiver_codes)
        receiver_states = self.read_in_attention(
            receiver_states, receiver_signatures, receiver_codes, sender_states
        )

        receiver_states = self.ffn(receiver_states, receiver_codes)

        return receiver_states


class ProjectionReadIn(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        code_dim: int,
        path_drop_prob: float = 0.0,
        ffn_layer_scale_initial_value: float = 1e-6,
        ffn_capacity_factor: float = 1.0,
        code_noise_scale: Optional[float] = None,
        use_geglu: bool = False,
        ffn_kwargs: Optional[dict] = None,
        mod_fc_cls: str = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
        ffn_mod_fc_cls: Optional[str] = None,
        ffn_mod_fc_kwargs: Optional[dict] = None,
    ):
        super(ProjectionReadIn, self).__init__()
        mod_fc_cls_ = getattr(utils, mod_fc_cls)
        self.read_in_projection = mod_fc_cls_(
            input_dim, state_dim, code_dim, **(mod_fc_kwargs or {})
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
        if code_noise_scale is not None:
            self.code_noiser = VectorScaleNoise(code_noise_scale)
        else:
            self.code_noiser = None

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_signatures: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_signatures: Optional[torch.Tensor] = None,
        sender_codes: Optional[torch.Tensor] = None,
    ):
        if self.code_noiser is not None:
            receiver_codes = self.code_noiser(receiver_codes)
        # Need to make sure that the number of receivers is the same as the number
        # of senders
        assert receiver_codes.shape[0] == sender_states.shape[1]
        receiver_states = self.read_in_projection(sender_states, receiver_codes)

        receiver_states = self.ffn(receiver_states, receiver_codes)

        return receiver_states


class Propagator(nn.Module):
    def __init__(
        self,
        state_dim: int,
        code_dim: int,
        num_heads: int = 1,
        head_dim: int = 64,
        path_drop_prob: float = 0.0,
        layer_scale_initial_value: float = 1e-6,
        ffn_capacity_factor: float = 1.0,
        latent_attention_kwargs: Optional[dict] = None,
        code_noise_scale: Optional[float] = None,
        include_residual_in_latent_attention: bool = True,
        use_geglu: bool = False,
        qkv_bias_in_attention: bool = True,
        ffn_kwargs: Optional[dict] = None,
        mod_fc_cls: Optional[str] = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
        ffn_mod_fc_cls: Optional[str] = None,
        ffn_mod_fc_kwargs: Optional[dict] = None,
        stitch_dim: Optional[int] = None,
    ):
        super(Propagator, self).__init__()
        self.latent_attention = KernelModulatedLatentAttention(
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
            ))
        if code_noise_scale is not None:
            self.code_noiser = VectorScaleNoise(code_noise_scale)
        else:
            self.code_noiser = None

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_signatures: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_signatures: Optional[torch.Tensor] = None,
        sender_codes: Optional[torch.Tensor] = None,
    ):
        if self.code_noiser is not None:
            # Add some noise to the code for moar exploration
            receiver_codes = self.code_noiser(receiver_codes)
            sender_codes = self.code_noiser(sender_codes)
        receiver_states = self.latent_attention(
            receiver_states,
            receiver_signatures,
            receiver_codes,
            sender_states,
            sender_signatures,
            sender_codes,
        )
        receiver_states = self.ffn(receiver_states, receiver_codes)
        return receiver_states

    @staticmethod
    def share_ffn_between_propagators(propagators: List["Propagator"]):
        ffn = propagators[0].ffn
        for propagator in propagators[1:]:
            propagator.ffn = ffn
        return propagators


class ReadOut(Propagator):
    def __init__(
        self,
        output_dim: Optional[int] = None,
        pre_output_layernorm: bool = True,
        use_head: bool = True,
        **super_kwargs,
    ):
        super(ReadOut, self).__init__(**super_kwargs)
        self.state_dim = super_kwargs["state_dim"]
        self.code_dim = super_kwargs["code_dim"]
        self.output_dim = output_dim
        self.pre_output_layernorm = pre_output_layernorm
        self.use_head = use_head
        # Modules
        mod_fc_cls = getattr(utils, super_kwargs.get("mod_fc_cls", "ModFC"))
        mod_fc_kwargs = super_kwargs.get("mod_fc_kwargs", None)
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

    def forward(
        self,
        receiver_states: torch.Tensor,
        receiver_signatures: torch.Tensor,
        receiver_codes: torch.Tensor,
        sender_states: torch.Tensor,
        sender_signatures: Optional[torch.Tensor] = None,
        sender_codes: Optional[torch.Tensor] = None,
    ):
        receiver_states = super(ReadOut, self).forward(
            receiver_states,
            receiver_signatures,
            receiver_codes,
            sender_states,
            sender_signatures,
            sender_codes,
        )
        receiver_states = self.head(receiver_states, receiver_codes)
        return receiver_states
