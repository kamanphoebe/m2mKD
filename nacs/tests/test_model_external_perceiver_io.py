import pytest
import yaml

BASE_CONFIG = """
model:
  name: PerceiverIO
  kwargs:
    depth: 8
    dim: 3
    queries_dim: 384
    logits_dim: 200
    num_latents: 320
    latent_dim: 384
    cross_heads: 1
    latent_heads: 6
    cross_dim_head: 64
    latent_dim_head: 64
    weight_tie_layers: false
    decoder_ff: true
    as_classifier: true
    from_bchw: false
    use_tokenizer: true
    tokenizer_kwargs:
      capacity_preset: femto
      positional_grid_kwargs:
        dim: 64
        grid_size: [4, 4]
"""


def test_perceiver_io():
    from addict import Addict

    import torch
    from neural_compilers.model import PerceiverIO

    config = Addict(yaml.safe_load(BASE_CONFIG))
    model = PerceiverIO(**config.model.kwargs)

    x = torch.randn(1, 3, 64, 64)
    y = model(x)

    assert list(y.shape) == [1, 200]
