import pytest
import yaml

BASE_CONFIG = """
model:
  name: PerceiverIO
  kwargs:
    depth: 8
    queries_dim: 384
    logits_dim: 2
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
    tokenizer_name: ClipTokenizer
    tokenizer_kwargs:
      text_dim: 512
      image_dim: 768
      num_text_tokens: 64
      num_image_tokens: 50
      projection_dim: 256 
      position_dim: 64
      category_dim: 64
    """


def test_perceiver_io_with_clip():
    from addict import Addict
    import torch
    from neural_compilers.model import PerceiverIO

    config = Addict(yaml.safe_load(BASE_CONFIG))
    config.model.kwargs.tokenizer_kwargs.use_pretrained_clip = True

    model = PerceiverIO(**config.model.kwargs)
    x = {
        "lhs_images": [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))],
        "rhs_images": [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))],
        "text": [
            "This is a cat.",
            "This is a big coconut dog.",
        ],
    }
    y = model(x)

    assert list(y.shape) == [2, 2]

    pass
