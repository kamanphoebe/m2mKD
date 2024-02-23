import pytest


def test_latent_graph():
    import torch
    from neural_compilers.model.nc.block import LatentGraph
    from addict import Addict

    config = Addict()
    config.state_dim = 64
    config.input_dim = 32
    config.code_dim = 128
    config.signature_dim = 48
    config.num_input_latents = 10
    config.num_mediator_latents = 20
    config.num_output_latents = 1
    config.num_iterations = 3
    config.output_dim = 16
    config.share_propagator_weights = False
    config.num_heads = 4
    config.head_dim = 32
    config.path_drop_prob = 0.1
    config.layer_scale_initial_value = 2e-6
    config.ffn_capacity_factor = 2.0
    config.code_noise_scale = 0.05
    config.use_code_noise_in_latent_seeder = True
    config.read_in_kwargs.read_in_layer_scale_initial_value = 0.5
    config.propagator_kwargs.latent_attention_kwargs.kernel_kwargs.truncation = 1.0
    config.propagator_layerwise_kwargs.path_drop_prob = [0.1, 0.2, 0.3]
    config.read_out_kwargs = None
    config.mediator_latent_kwargs.graph_preset = "watts-strogatz"

    lg = LatentGraph(**config)

    batch_size = 2
    num_inputs = 30
    inputs = torch.randn(batch_size, num_inputs, config.input_dim)

    outputs = lg(inputs)

    def check(key, shape):
        assert list(outputs[key].shape) == list(shape)

    check("input_states", [batch_size, config.num_input_latents, config.state_dim])
    check("mediator_states", [batch_size, config.num_mediator_latents, config.state_dim])
    check("output_states", [batch_size, config.num_output_latents, config.output_dim])


if __name__ == '__main__':
    pytest.main()
