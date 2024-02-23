from typing import Optional, Tuple, Union, Dict, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as eo
from torch.distributions import Beta, Uniform
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import (
    CLIPTokenizer,
    CLIPModel,
    FlaxCLIPVisionModel,
    LxmertTokenizer,
    LxmertModel,
    BertTokenizer,
)

from neural_compilers.utils import drop_path, ModFC, to_device, batch_dicts


class DropPath(nn.Module):
    """
    Implements stochastic depth, partially swiped from timm.
    Assumes that new = old + f(old), and applies new = old + drop(f(old)).
    """

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(
        self,
        old: torch.Tensor,
        new: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ):
        if self.drop_prob == 0.0:
            # This code-path essentially acts like identity
            if new is not None:
                return new
            assert residual is not None
            return old + residual
        # We're dropping paths, so this is not the identity
        return drop_path(
            drop_prob=self.drop_prob,
            old=old,
            new=new,
            residual=residual,
            training=self.training,
        )


class LayerScale(nn.Module):
    def __init__(self, dim: int, initial_value: Optional[float] = 1e-6):
        super(LayerScale, self).__init__()
        self.dim = dim
        if initial_value is not None:
            self.gamma = nn.Parameter(torch.ones(dim).mul_(initial_value))
        else:
            self.gamma = None

    def forward(self, input: torch.Tensor):
        if self.gamma is None:
            # No-op
            return input
        # input.shape = ...C
        # Broadcasting takes care of things
        return self.gamma * input


class PositionalGrid(nn.Module):
    def __init__(self, dim: int, grid_size: Tuple[int, int] = (8, 8)):
        super(PositionalGrid, self).__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.grid = nn.Parameter(torch.randn(dim, *grid_size))

    def forward(
        self,
        batch_size: int,
        spatial_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        # Add the batch axis
        batched_grid = eo.repeat(self.grid, "c h w -> b c h w", b=batch_size)
        if spatial_size is not None:
            spatial_size = (
                (spatial_size, spatial_size)
                if isinstance(spatial_size, int)
                else spatial_size
            )
            assert len(spatial_size) == 2
            spatial_size = self.grid_size
        # Interpolate only if it's required
        if tuple(spatial_size) != tuple(self.grid_size):
            interpolated_grid = F.adaptive_avg_pool2d(
                batched_grid, output_size=spatial_size
            )
        else:
            interpolated_grid = batched_grid
        return interpolated_grid


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features):
        super(RandomFourierFeatures, self).__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn(out_features // 2, in_features))
        # noinspection PyArgumentList
        self.register_buffer(
            "bias", torch.empty(out_features // 2).uniform_(0, 2 * 3.1415926535)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.einsum("oi,...i->...o", self.weight, input) + self.bias
        output = torch.cat([output.cos(), output.sin()], dim=-1)
        return output


class VanillaPositionalEncodings(nn.Module):
    def __init__(self, dim: int, num_positions: int):
        super(VanillaPositionalEncodings, self).__init__()
        self.dim = dim
        self.num_positions = num_positions
        # Parameters
        self.embeddings = nn.Parameter(torch.randn(1, self.num_positions, self.dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input.shape = BUC
        output = input + self.embeddings
        return output


class VectorMixup(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super(VectorMixup, self).__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    @torch.no_grad()
    def sample_alpha(self, num_samples: int) -> torch.Tensor:
        # beta_samples.shape = N
        # beta_samples as of this line can be either close to zero or one, ...
        beta_samples = Beta(self.alpha, self.alpha).sample((num_samples,))
        # ... but while this makes sense for vanilla Mixup, it doesn't make
        # sense in the context of mixing up codes or signatures. Consequently,
        # we only want to keep the values close to 0.
        return torch.minimum(beta_samples, 1.0 - beta_samples)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input
        # input.shape = NC
        permuted_indices = torch.randperm(input.shape[0], device=input.device)
        permuted_input = input[permuted_indices]
        # alpha.shape = N1
        alpha = self.sample_alpha(input.shape[0])
        mixed_input = (1.0 - alpha[:, None]) * input + (alpha[:, None]) * permuted_input
        return mixed_input


class VectorScaleNoise(nn.Module):
    def __init__(self, alpha: float = 0.01):
        super(VectorScaleNoise, self).__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input
        # input.shape = NC
        with torch.no_grad():
            input_copy = input.clone()
            uniform_scale = input_copy.abs_().mul_(self.alpha)
            noise = Uniform(-uniform_scale, uniform_scale).sample()
        input = input + noise
        return input


class PretrainedCLIPBackbone(nn.Module):
    def __init__(self, cache_dir="hf_cache"):
        super(PretrainedCLIPBackbone, self).__init__()
        self.cache_dir = os.path.join(os.environ.get("SCRATCH", ""), cache_dir)
        self.register_buffer("device_inference", torch.tensor(0.0))
        self.backbone = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=self.cache_dir
        )
        self.vision_model = self.backbone.vision_model.to("cuda:0")
        self.vision = torch.nn.Conv2d(3, 768, 32, 32).cuda()

        self.processor = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=self.cache_dir,
        )
        self.boo = True

    def does_not_require_grad(self):
        for p in self.parameters():
            p.requires_grad = False
        return self

    def collate(self, batch):
        lhs_images, rhs_images, text = (
            batch["lhs_images"],
            batch["rhs_images"],
            batch["text"],
        )
        device = self.device_inference.device
        text = self.processor(
            text=list(text),
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            is_split_into_words=False,
        )

        inputs = {
            "pixel_values": torch.cat([lhs_images, rhs_images], dim=0),
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
        }
        inputs = to_device(inputs, device=device, non_blocking=True)

        return inputs

    def forward(self, data: Dict[str, List[torch.Tensor]]):
        inputs = self.collate(data)

        vision_model_output = self.vision_model(
            **{"pixel_values": inputs["pixel_values"]}
        )

        lhs_images, rhs_images = vision_model_output.get("last_hidden_state").chunk(
            2, dim=0
        )

        # Process the text
        text_model_output = self.backbone.text_model(
            **{
                "attention_mask": inputs["attention_mask"],
                "input_ids": inputs["input_ids"],
            }
        )
        text = text_model_output.last_hidden_state
        attention_mask = inputs["attention_mask"].bool()
        return dict(
            lhs_images=lhs_images,
            rhs_images=rhs_images,
            text=text,
            attention_mask=attention_mask,
        )


def get_student_state_dict(dir: str, name: str, student_type: str='tinynet'):
    student_state_dict = torch.load(os.path.join(dir, f'{name}.pth'))["model"]
    if 'MediatorStudent' in name:
        return student_state_dict
    tokenizer_dict = {}
    attention_dict = {}
    for key, item in student_state_dict.items():
        if 'positional_grid' in key and student_type == 'tinynet':
            continue
        if key.startswith('tokenizer.'):
            new_key = key.replace('tokenizer.', '')
            tokenizer_dict[new_key] = item
        else:
            new_key = key.replace('read_in.', '') # This line has no impact for ReadOutStudent 
            attention_dict[new_key] = item
    return tokenizer_dict, attention_dict
    