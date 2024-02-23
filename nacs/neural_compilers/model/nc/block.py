from copy import deepcopy
from typing import Optional
import os

import torch
import torch.nn as nn

from neural_compilers.model.nc.attention import (
    AttentiveReadIn,
    ProjectionReadIn,
    Propagator,
    ReadOut,
)
from neural_compilers.model.nc.seeding import Latents, LatentSeeder
from neural_compilers.model.nc.utils import get_student_state_dict, DropPath
from neural_compilers.utils import override, broadcast_update, keep_valid_backups, GEGLU


class LatentGraph(nn.Module):
    """
    This module covers attentive message passing in two steps:
        1. Read-in from inputs to the states of the input latents
        2. Propagate messages from the union of input and mediator latents
           to all mediator latents
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        code_dim: int,
        signature_dim: int,
        num_input_latents: int,
        num_mediator_latents: int,
        num_output_latents: int,
        num_iterations: int,
        output_dim: Optional[int] = None,
        *,
        share_propagator_weights: bool = True,
        share_ffn_between_propagators: bool = False,
        num_heads: int = 1,
        head_dim: int = 64,
        path_drop_prob: float = 0.0,
        layer_scale_initial_value: float = 1e-6,
        ffn_capacity_factor: float = 1.0,
        code_noise_scale: Optional[float] = None,
        learnable_signatures: bool = True,
        learnable_codes: bool = True,
        use_code_noise_in_latent_seeder: bool = False,
        share_codes_between_iterations: bool = True,
        enable_input_output_communication: bool = True,
        enable_persistent_input_mediator_communication: bool = True,
        disable_input_mediator_communication: bool = False,
        use_input_states_as_mediator_states: bool = False,
        use_geglu: bool = False,
        qkv_bias_in_attention: bool = True,
        mod_fc_cls: str = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
        ffn_mod_fc_cls: Optional[str] = None,
        ffn_mod_fc_kwargs: Optional[dict] = None,
        use_latent_seeder: bool = True,
        latent_seeder_kwargs: Optional[dict] = None,
        input_latent_kwargs: Optional[dict] = None,
        mediator_latent_kwargs: Optional[dict] = None,
        output_latent_kwargs: Optional[dict] = None,
        read_in_cls: str = "AttentiveReadIn",
        read_in_kwargs: Optional[dict] = None,
        propagator_kwargs: Optional[dict] = None,
        propagator_layerwise_kwargs: Optional[dict] = None,
        read_out_kwargs: Optional[dict] = None,
        student_state_dict_dir: Optional[str] = None,
        predefined_codes_dir: Optional[str] = None,
        stitch_ckpt_path: Optional[str] = None,
        stitch_dim: Optional[int] = None,
    ):
        super(LatentGraph, self).__init__()
        # Save what needs to be saved
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.signature_dim = signature_dim
        self.num_input_latents = num_input_latents
        self.num_mediator_latents = num_mediator_latents
        self.num_output_latents = num_output_latents
        self.output_dim = output_dim
        self.share_propagator_weights = share_propagator_weights
        self.share_ffn_between_propagators = share_ffn_between_propagators
        self.num_iterations = num_iterations
        self.share_codes_between_iterations = share_codes_between_iterations
        self.enable_input_output_communication = enable_input_output_communication
        self.enable_persistent_input_mediator_communication = (
            enable_persistent_input_mediator_communication
        )
        self.disable_input_mediator_communication = disable_input_mediator_communication
        self.use_input_states_as_mediator_states = use_input_states_as_mediator_states
        self.use_geglu = use_geglu
        self.qkv_bias_in_attention = qkv_bias_in_attention
        self.use_latent_seeder = use_latent_seeder
        self.stitch_pairs = []
        if stitch_ckpt_path:
            stitch_ckpt = torch.load(stitch_ckpt_path)
            self.stitch_pairs = nn.ModuleList()
            for pair in stitch_ckpt:
                self.stitch_pairs.append(nn.ModuleList())
                self.stitch_pairs[-1].append(nn.Linear(state_dim, stitch_dim))
                self.stitch_pairs[-1].append(nn.Linear(stitch_dim, state_dim))
                self.stitch_pairs[-1][0].load_state_dict(pair['post_stitch'])
                self.stitch_pairs[-1][1].load_state_dict(pair['pre_stitch'])
            self.stitch_norm = nn.LayerNorm(state_dim)
            self.stitch_drop = DropPath(path_drop_prob)
            self.stitch_act = nn.GELU()

        # Initialize the latents
        self.input_latents = Latents(
            **override(
                input_latent_kwargs,
                num_latents=num_input_latents,
                signature_dim=signature_dim,
                code_dim=code_dim,
                state_dim=state_dim,
                learnable_signatures=learnable_signatures,
                learnable_codes=learnable_codes,
                learnable_initial_states=(not use_latent_seeder),
                predefined_codes_path=os.path.join(predefined_codes_dir, 'readin_codes.pth') if predefined_codes_dir else None,
            ),
        )
        self.mediator_latents = Latents(
            **override(
                mediator_latent_kwargs,
                num_latents=num_mediator_latents,
                signature_dim=signature_dim,
                code_dim=code_dim,
                state_dim=state_dim,
                learnable_signatures=learnable_signatures,
                learnable_codes=learnable_codes,
                learnable_initial_states=(not use_latent_seeder),
                num_layers=(
                    None if self.share_codes_between_iterations else self.num_iterations
                ),
                predefined_codes_path=os.path.join(predefined_codes_dir, 'mediator_codes.pth') if predefined_codes_dir else None,
            ),
        )
        self.output_latents = Latents(
            **override(
                output_latent_kwargs,
                num_latents=num_output_latents,
                signature_dim=signature_dim,
                code_dim=code_dim,
                state_dim=state_dim,
                learnable_signatures=learnable_signatures,
                learnable_codes=learnable_codes,
                learnable_initial_states=(not use_latent_seeder),
                predefined_codes_path=os.path.join(predefined_codes_dir, 'readout_codes.pth') if predefined_codes_dir else None,
            ),
        )
        # Initialize the seeder
        if self.use_latent_seeder:
            latent_seeder_kwargs = override(
                latent_seeder_kwargs,
                state_dim=state_dim,
                code_dim=code_dim,
                **(
                    dict(code_noise_scale=code_noise_scale)
                    if use_code_noise_in_latent_seeder
                    else {}
                ),
            )
            self.latent_seeder = LatentSeeder(**latent_seeder_kwargs)
        else:
            self.latent_seeder = None
        # Initialize the read-in
        read_in_cls = {
            "AttentiveReadIn": AttentiveReadIn,
            "ProjectionReadIn": ProjectionReadIn,
        }[read_in_cls]
        self.read_in = read_in_cls(
            **keep_valid_backups(
                read_in_cls,
                read_in_kwargs,
                input_dim=input_dim,
                state_dim=state_dim,
                code_dim=code_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                path_drop_prob=path_drop_prob,
                ffn_layer_scale_initial_value=layer_scale_initial_value,
                ffn_capacity_factor=ffn_capacity_factor,
                code_noise_scale=code_noise_scale,
                use_geglu=use_geglu,
                qkv_bias_in_attention=qkv_bias_in_attention,
                mod_fc_cls=mod_fc_cls,
                mod_fc_kwargs=mod_fc_kwargs,
                ffn_mod_fc_cls=ffn_mod_fc_cls,
                ffn_mod_fc_kwargs=ffn_mod_fc_kwargs,
            ),
        )
        if student_state_dict_dir:
            _, readin_state_dict = get_student_state_dict(student_state_dict_dir, 'ReadInStudent')
            self.read_in.load_state_dict(readin_state_dict, strict=False)
        # Initialize the propagator
        # FIXME: If we're not sharing weights, the kernels in each propagator may
        #  have a different learnable bandwidth. This might not be what we want.
        # TODO: It might make sense to have different truncations for input and
        #  output latents, vs. mediator latents.
        # All this gymnastics is to be able to configure each propagator
        # differently (e.g. to have different path drop for different layers).
        num_unique_propagators = (
            1 if self.share_propagator_weights else self.num_iterations
        )
        propagator_kwargs = override(
            propagator_kwargs,
            state_dim=state_dim,
            code_dim=code_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            path_drop_prob=path_drop_prob,
            layer_scale_initial_value=layer_scale_initial_value,
            ffn_capacity_factor=ffn_capacity_factor,
            code_noise_scale=code_noise_scale,
            use_geglu=use_geglu,
            qkv_bias_in_attention=qkv_bias_in_attention,
            mod_fc_cls=mod_fc_cls,
            mod_fc_kwargs=mod_fc_kwargs,
            ffn_mod_fc_cls=ffn_mod_fc_cls,
            ffn_mod_fc_kwargs=ffn_mod_fc_kwargs,
        )
        if propagator_layerwise_kwargs is not None:
            broadcasted_propagator_kwargs = broadcast_update(
                propagator_kwargs, propagator_layerwise_kwargs, num_unique_propagators
            )
        else:
            broadcasted_propagator_kwargs = [
                deepcopy(propagator_kwargs) for _ in range(num_unique_propagators)
            ]
        assert len(broadcasted_propagator_kwargs) == num_unique_propagators
        propagators = [
            Propagator(**_propagator_kwargs)
            for _propagator_kwargs in broadcasted_propagator_kwargs
        ]
        if student_state_dict_dir:
            for i, propagator in enumerate(propagators, start=1):
                mediator_state_dict = get_student_state_dict(student_state_dict_dir, f'MediatorStudent_{i}')
                propagator.load_state_dict(mediator_state_dict, strict=False)
        if len(propagators) == 1:
            # Note that this doesn't copy the propagators, but only makes
            # a shallow copy. Parameters of these copies are still shared.
            propagators = propagators * self.num_iterations
        if share_ffn_between_propagators:
            propagators = Propagator.share_ffn_between_propagators(propagators)
        self.propagators = nn.ModuleList(propagators)
        # Initialize the readout
        self.read_out = ReadOut(
            **override(
                read_out_kwargs,
                state_dim=state_dim,
                code_dim=code_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                output_dim=output_dim,
                path_drop_prob=path_drop_prob,
                layer_scale_initial_value=layer_scale_initial_value,
                ffn_capacity_factor=ffn_capacity_factor,
                code_noise_scale=code_noise_scale,
                use_geglu=use_geglu,
                qkv_bias_in_attention=qkv_bias_in_attention,
                mod_fc_cls=mod_fc_cls,
                mod_fc_kwargs=mod_fc_kwargs,
                ffn_mod_fc_cls=ffn_mod_fc_cls,
                ffn_mod_fc_kwargs=ffn_mod_fc_kwargs,
            ),
        )
        if student_state_dict_dir:
            _, readout_state_dict = get_student_state_dict(student_state_dict_dir, 'ReadOutStudent')
            self.read_out.load_state_dict(readout_state_dict, strict=False)

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_states: Optional[torch.Tensor] = None,
        mediator_states: Optional[torch.Tensor] = None,
        output_states: Optional[torch.Tensor] = None,
    ):
        if inputs is not None:
            # input.shape = BVC
            batch_size = inputs.shape[0]
        else:
            # inputs is None, but we need to infer the batch size
            if input_states is not None:
                batch_size = input_states.shape[0]
            elif mediator_states is not None:
                batch_size = mediator_states.shape[0]
            elif output_states is not None:
                batch_size = output_states.shape[0]
            else:
                raise RuntimeError("batch size could not be inferred.")
        # --------------------
        # Get the initial states if required
        if input_states is None:
            if self.latent_seeder is not None:
                # We're using a latent-seeder to convert codes to seed
                input_states = self.latent_seeder(self.input_latents.codes, batch_size)
            else:
                # We're learning a separate initial state per latent
                input_states = self.input_latents.get_initial_states(batch_size)
        if mediator_states is None:
            # We don't need to use the latent seeder if we'll be setting
            # the mediator states to the input states later anyway.
            if not self.use_input_states_as_mediator_states:
                if self.latent_seeder is not None:
                    mediator_states = self.latent_seeder(
                        self.mediator_latents.get_code(0), batch_size
                    )
                else:
                    mediator_states = self.mediator_latents.get_initial_states(
                        batch_size
                    )
        if output_states is None:
            if self.latent_seeder is not None:
                output_states = self.latent_seeder(
                    self.output_latents.codes, batch_size
                )
            else:
                output_states = self.output_latents.get_initial_states(batch_size)
        # --------------------
        # Read-in inputs
        if inputs is not None:
            assert self.read_in is not None
            input_states = self.read_in(
                # Receiver -->
                input_states,
                self.input_latents.signatures,
                self.input_latents.codes,
                # Sender -->
                inputs,
            )
        else:
            assert input_states is not None
        if self.stitch_pairs:
            residual = input_states
            input_states = self.stitch_norm(input_states)
            input_states = self.stitch_pairs[0][0](input_states)
            input_states = self.stitch_act(input_states)
            input_states = self.stitch_pairs[0][1](input_states)
            input_states = self.stitch_drop(old=residual, residual=input_states)
            # input_states = input_states + residual
        # If required, set the mediator states to input states with a check
        if self.use_input_states_as_mediator_states:
            assert self.mediator_latents.num_latents == self.input_latents.num_latents
            mediator_states = input_states
        assert mediator_states is not None
        # --------------------
        # Next step is to pass messages from the input and mediator latents
        # to the mediator latents.
        input_and_mediator_signatures = torch.cat(
            [self.input_latents.signatures, self.mediator_latents.signatures], dim=0
        )
        for depth, propagator in enumerate(self.propagators):
            if (
                depth > 0 and not self.enable_persistent_input_mediator_communication
            ) or self.disable_input_mediator_communication:
                # In this code-path, the input latents don't feed in to the graph at
                # every step (depth).
                propagator_sender_states = mediator_states
                propagator_sender_signatures = self.mediator_latents.signatures
                propagator_sender_codes = self.mediator_latents.get_code(depth)
            else:
                # In this code-path, the input latents feed in to the graph at every
                # step (depth). This code-path is always active at depth = 0 to
                # introduce the input latents to the mediators.
                propagator_sender_states = torch.cat(
                    [input_states, mediator_states], dim=1
                )
                propagator_sender_signatures = input_and_mediator_signatures
                propagator_sender_codes = torch.cat(
                    [
                        self.input_latents.codes,
                        self.mediator_latents.get_code(depth),
                    ],
                    dim=0,
                )
            mediator_states = propagator(
                # Receiver -->
                mediator_states,
                self.mediator_latents.signatures,
                self.mediator_latents.get_code(depth),
                # Sender -->
                propagator_sender_states,
                propagator_sender_signatures,
                propagator_sender_codes,
            )
            if self.stitch_pairs:
                residual = mediator_states
                mediator_states = self.stitch_norm(mediator_states)
                mediator_states = self.stitch_pairs[depth+1][0](mediator_states)
                mediator_states = self.stitch_act(mediator_states)
                mediator_states = self.stitch_pairs[depth+1][1](mediator_states)
                mediator_states = self.stitch_drop(old=residual, residual=mediator_states)
                # mediator_states = mediator_states + residual
        # --------------------
        # The final step is to pass messages from mediator and optionally input latents to
        # output latents
        if self.enable_input_output_communication:
            # In this code-path, both input and mediators send messages to the output latents.
            read_out_sender_states = torch.cat([input_states, mediator_states], dim=1)
            read_out_sender_signatures = input_and_mediator_signatures
            read_out_sender_codes = torch.cat(
                [
                    self.input_latents.codes,
                    self.mediator_latents.get_code(-1),
                ],
                dim=0,
            )
        else:
            # In this code-path, only the mediators send messages to the output latents
            read_out_sender_states = mediator_states
            read_out_sender_signatures = self.mediator_latents.signatures
            read_out_sender_codes = self.mediator_latents.get_code(-1)
        output_states = self.read_out(
            # Receiver -->
            output_states,
            self.output_latents.signatures,
            self.output_latents.codes,
            # Sender -->
            read_out_sender_states,
            read_out_sender_signatures,
            read_out_sender_codes,
        )
        # --------------------
        # Done!
        return dict(
            input_states=input_states,
            mediator_states=mediator_states,
            output_states=output_states,
        )
