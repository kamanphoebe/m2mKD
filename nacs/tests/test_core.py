import pytest
import yaml


BASE_CONFIG = """
model:
  name: NeuralCompiler
  kwargs:
    input_dim: 3
    output_dim: 1000
    state_dim: 384
    input_tokenizer_type: "ConvNeXtImageTokenizer"
    input_tokenizer_kwargs:
      capacity_preset: femto
      num_stages: 2
      positional_grid_kwargs:
        dim: 64
        grid_size: [8, 8]
    latent_graph_type: "LatentGraph"
    latent_graph_kwargs:
      code_dim: 384
      signature_dim: 64
      num_input_latents: 64
      num_mediator_latents: 256
      num_output_latents: 64
      num_iterations: 8
      share_propagator_weights: false
      share_codes_between_iterations: false
      share_ffn_between_propagators: true
      num_heads: 6
      head_dim: 64
      path_drop_prob: 0.1
      layer_scale_initial_value: 0.1
      ffn_capacity_factor: 1.3333
      enable_input_output_communication: false
      enable_persistent_input_mediator_communication: false
      use_latent_seeder: true
      input_latent_kwargs: null
      mediator_latent_kwargs:
        graph_preset: watts-strogatz
        graph_generator_kwargs:
          truncation: 0.9
      output_latent_kwargs: null
      read_in_cls: AttentiveReadIn
      read_in_kwargs:
        path_drop_prob: 0.0
      propagator_kwargs:
        latent_attention_kwargs:
          kernel_type: "DotProductKernel"
          kernel_kwargs:
            truncation: 0.9
            stochastic_kernel: true
      read_out_kwargs: 
        use_head: false
        latent_attention_kwargs: 
          kernel_type: "DotProductKernel"
          kernel_kwargs:
            truncation: 0.9
            stochastic_kernel: true
      mod_fc_cls: FC
      ffn_mod_fc_cls: ModFC
    output_tokenizer_type: "OutputLatentPooling"
    output_tokenizer_kwargs: 
      softmax_type: "gumbel-hard"
    simplified_interface: true
"""


PERCEIVER_LIKE_CONFIG = """
model:
  name: NeuralCompiler
  kwargs:
    input_dim: 3
    output_dim: 1000
    state_dim: 384
    input_tokenizer_type: "ConvNeXtImageTokenizer"
    input_tokenizer_kwargs:
      capacity_preset: femto
      positional_grid_kwargs:
        dim: 64
        grid_size: [4, 4]
    latent_graph_type: "LatentGraph"
    latent_graph_kwargs:
      code_dim: 384
      signature_dim: 64
      num_input_latents: 320
      num_mediator_latents: 320
      num_output_latents: 1
      num_iterations: 8
      share_propagator_weights: false
      num_heads: 6
      head_dim: 64
      path_drop_prob: 0.0
      layer_scale_initial_value: null
      ffn_capacity_factor: 4
      code_noise_scale: null
      use_code_noise_in_latent_seeder: false
      disable_input_mediator_communication: true
      use_input_states_as_mediator_states: true
      enable_input_output_communication: false
      latent_seeder_kwargs:
        use_state_noise: false
        use_code_as_mean_state: true
      input_latent_kwargs:
        graph_preset: complete
        learnable_signatures: false
      mediator_latent_kwargs:
        graph_preset: complete
        learnable_signatures: false
        learnable_codes: false
        init_with_identical_codes: true
      output_latent_kwargs:
        graph_preset: complete
        learnable_signatures: false
      read_in_kwargs:
        path_drop_prob: 0.0
        num_heads: 1
        read_in_layer_scale_initial_value: null
        include_residual_in_read_in_attention: true
        read_in_attention_kwargs:
          qkv_bias: false
        ffn_kwargs: 
          use_geglu: true
      propagator_kwargs:
        latent_attention_kwargs:
          share_layernorm: true
          mask_attn_scores_with_affinities: false
          qkv_bias: false
          kernel_type: "DotProductKernel"
          kernel_kwargs:
            truncation: 0.9
        ffn_kwargs:
          use_geglu: true
      read_out_kwargs:
        num_heads: 1
        include_residual_in_latent_attention: false
        pre_output_layernorm: false
        latent_attention_kwargs:
          mask_attn_scores_with_affinities: false
          share_layernorm: false
          qkv_bias: false
        ffn_kwargs:
          use_geglu: true
      mod_fc_cls: FC
    output_tokenizer_type: "FirstSlotAsLogits"
    output_tokenizer_kwargs: null
    simplified_interface: true
"""


def test_neural_compiler():
    from addict import Addict
    import torch
    from neural_compilers.model import NeuralCompiler

    image_size = 64
    config = BASE_CONFIG
    config = Addict(yaml.safe_load(config))

    model = NeuralCompiler(**config.model.kwargs)

    image = torch.randn(1, 3, image_size, image_size)
    output = model(image)

    assert list(output.shape) == [1, 1000]


if __name__ == '__main__':
    pytest.main()