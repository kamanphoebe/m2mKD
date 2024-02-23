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
      capacity_preset: tiny
      positional_grid_kwargs:
        dim: 64
        grid_size: [14, 14]
    latent_graph_type: "LatentGraph"
    latent_graph_kwargs:
      code_dim: 384
      signature_dim: 64
      num_input_latents: 64
      num_mediator_latents: 256
      num_output_latents: 1
      num_iterations: 8
      share_propagator_weights: false
      num_heads: 6
      head_dim: 64
      path_drop_prob: 0.1
      layer_scale_initial_value: 0.1
      ffn_capacity_factor: 1.3333
      input_latent_kwargs: null
      mediator_latent_kwargs:
        graph_preset: watts-strogatz
        graph_generator_kwargs:
          truncation: 0.9
      output_latent_kwargs: null
      read_in_kwargs:
        path_drop_prob: 0.0
      propagator_kwargs:
        latent_attention_kwargs:
          kernel_type: "DotProductKernel"
          kernel_kwargs:
            truncation: 0.9
      read_out_kwargs: null
    output_tokenizer_type: "FirstSlotAsLogits"
    output_tokenizer_kwargs: null
    simplified_interface: true
"""


def test_signature_hinge_repulsion():
    from addict import Addict
    from neural_compilers.model import NeuralCompiler
    from neural_compilers.objectives.regularization import SignatureHingeRepulsion

    config = Addict(yaml.safe_load(BASE_CONFIG))
    model = NeuralCompiler(**config.model.kwargs)
    loss_fn = SignatureHingeRepulsion(model, 0.6)

    loss = loss_fn()
    loss.backward()

    assert model.latent_graphs[0].mediator_latents.signatures.grad is not None
    assert model.latent_graphs[0].input_latents.signatures.grad is not None
    assert model.latent_graphs[0].output_latents.signatures.grad is not None


def test_signature_distribution_regularization():
    from addict import Addict
    from neural_compilers.model import NeuralCompiler
    from neural_compilers.objectives.regularization import (
        SignatureDistributionRegularization,
    )

    config = Addict(yaml.safe_load(BASE_CONFIG))
    model = NeuralCompiler(**config.model.kwargs)
    loss_fn = SignatureDistributionRegularization(model, num_components=5)

    loss = loss_fn()
    loss.backward()

    assert model.latent_graphs[0].mediator_latents.signatures.grad is not None
    assert model.latent_graphs[0].input_latents.signatures.grad is not None
    assert model.latent_graphs[0].output_latents.signatures.grad is not None


def test_stochastic_block_model_regularization():
    from addict import Addict
    from neural_compilers.model import NeuralCompiler
    from neural_compilers.objectives.regularization import (
        StochasticBlockModelRegularization,
    )

    config = Addict(yaml.safe_load(BASE_CONFIG))
    model = NeuralCompiler(**config.model.kwargs)
    loss_fn = StochasticBlockModelRegularization(
        model,
        num_clusters=8,
        regularize_input_latents=False,
        regularize_mediator_latents=True,
        regularize_output_latents=False,
    )

    loss = loss_fn()
    loss.backward()

    assert model.latent_graphs[0].input_latents.signatures.grad is None
    assert model.latent_graphs[0].mediator_latents.signatures.grad is not None
    assert model.latent_graphs[0].output_latents.signatures.grad is None


if __name__ == "__main__":
    pytest.main()
