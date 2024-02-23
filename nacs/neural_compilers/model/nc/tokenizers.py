from typing import Optional, Dict

import torch
import torch.nn as nn
import einops as eo
from torch.nn.functional import gumbel_softmax

from neural_compilers.model.external.convnext import ConvNeXt
from neural_compilers.model.nc.utils import PositionalGrid, PretrainedCLIPBackbone
from neural_compilers.utils import override


class InputTokenizer(nn.Module):
    @property
    def output_dim(self):
        raise NotImplementedError


class ClipTokenizer(InputTokenizer):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        num_image_tokens: int,
        num_text_tokens: int,
        projection_dim: int,
        position_dim: int,
        category_dim: int,
        use_pretrained_clip: bool = False,
    ):
        super().__init__()
        # Attributes
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.projection_dim = projection_dim
        self.position_dim = position_dim
        self.num_image_tokens = num_image_tokens
        self.num_text_tokens = num_text_tokens
        self.category_dim = category_dim
        # Modules
        self.backbone = None
        if use_pretrained_clip:
            self.backbone = PretrainedCLIPBackbone().does_not_require_grad()
        self.text_embedding = nn.Linear(text_dim, projection_dim)
        self.image_embedding = nn.Linear(image_dim, projection_dim)
        self.image_position_embedding = nn.Embedding(num_image_tokens, position_dim)
        self.text_position_embedding = nn.Embedding(num_text_tokens, position_dim)
        # There are 3 categories -- left image, right image and text
        self.category_embedding = nn.Embedding(3, category_dim)

    def _evaluate_backbone(self, data: Dict[str, torch.Tensor]):
        if self.backbone is None:
            assert "image" in data and "text" in data and "mask" in data
            return data["image"], data["text"], data["mask"]
        else:
            with torch.inference_mode():
                data = self.backbone(data)
            # image.shape = B2NC
            image = torch.stack([data["rhs_images"], data["lhs_images"]], dim=1)
            # mask.shape = B(2N + M)
            B, N = image.shape[0], image.shape[2]
            attention_mask = data["attention_mask"]
            mask = torch.cat(
                [
                    torch.ones(
                        B, 2 * N, dtype=torch.bool, device=attention_mask.device
                    ),
                    attention_mask,
                ],
                dim=1,
            )
            return [
                image.detach().clone(),
                data["text"].detach().clone(),
                mask.detach().clone(),
            ]

    def forward(self, data: Dict[str, torch.Tensor]):

        image, text, mask = self._evaluate_backbone(data)
        # with record_function("embed and cat"):
        # text.shape = BMC where M is the number of tokens
        # image.shape = B2NC where N is the number of patches
        batch_size = text.shape[0]
        num_text_tokens = text.shape[1]
        num_image_tokens = image.shape[2]
        device = image.device
        assert num_text_tokens == self.num_text_tokens
        assert num_image_tokens == self.num_image_tokens
        # ----- Text and image embeddings -----
        # text_embedding.shape = BMC
        text_embedding = self.text_embedding(text)
        # image_embedding.shape = B2NC
        image_embedding = self.image_embedding(image)
        # stacked_image is just the two images flattened out
        stacked_image = eo.rearrange(image_embedding, "b p n c -> b (p n) c")
        # embedding.shape = B(2N + M)C
        embeddings = torch.cat([stacked_image, text_embedding], dim=1)
        # ----- Positional embeddings -----
        # image_position_embedding.shape = NC
        image_position_embedding = self.image_position_embedding(
            torch.arange(num_image_tokens, dtype=torch.long, device=device)
        )
        # text_position_embedding.shape = MC
        text_position_embedding = self.text_position_embedding(
            torch.arange(num_text_tokens, dtype=torch.long, device=device)
        )
        # position_embedding.shape = B(2N + M)C
        position_embeddings = torch.cat(
            [
                eo.repeat(
                    image_position_embedding,
                    "n c -> b (p n) c",
                    p=2,
                    b=batch_size,
                ),
                eo.repeat(text_position_embedding, "m c -> b m c", b=batch_size),
            ],
            dim=1,
        )
        # ----- Category embeddings -----
        # First N: category 0, next N: category 1, next M: category 2
        # categories.shape = (2N + M)
        categories = torch.cat(
            [
                torch.zeros(num_image_tokens, device=device, dtype=torch.long),
                torch.zeros(num_image_tokens, device=device, dtype=torch.long).add_(1),
                torch.zeros(num_text_tokens, device=device, dtype=torch.long).add_(2),
            ],
            dim=0,
        )
        # category_embeddings.shape = B(2N + M)C
        category_embeddings = eo.repeat(
            self.category_embedding(categories), "n c -> b n c", b=batch_size
        )
        # ----- cat'em all -----
        embeddings = torch.cat(
            [embeddings, position_embeddings, category_embeddings], dim=-1
        )

        return dict(embeddings=embeddings, mask=mask)

    @property
    def output_dim(self):
        return self.projection_dim + self.position_dim + self.category_dim


class ConvNeXtImageTokenizer(InputTokenizer):
    PRESET_TO_DEPTHS = {
        "tiny": [3, 3, 9],
        "mini": [3, 3, 6],
        "micro": [3, 3, 3],
        "nano": [1, 1, 1],
        "pico": [1, 1, 0],
        "femto": [1, 0, 0],
        "atto": [0, 0, 0],
    }

    def __init__(
        self,
        input_dim: int = 3,
        repr_dim: int = 384,
        capacity_preset: str = "tiny",
        num_stages: int = 3,
        convnext_kwargs: Optional[dict] = None,
        positional_grid_kwargs: Optional[dict] = None,
    ):
        super(ConvNeXtImageTokenizer, self).__init__()
        depths = self.PRESET_TO_DEPTHS.get(capacity_preset, [3, 3, 27])[:num_stages]
        dims = [repr_dim // 4, repr_dim // 2, repr_dim][:num_stages]
        self.backbone = ConvNeXt(
            **override(
                convnext_kwargs or {},
                num_classes=None,
                gap=False,
                in_chans=input_dim,
                depths=depths,
                dims=dims,
            )
        )
        if positional_grid_kwargs is not None:
            self.positional_grid = PositionalGrid(**positional_grid_kwargs)
        else:
            self.positional_grid = None

    def forward(self, input: torch.Tensor):
        # input.shape = BCHW
        # features.shape = BCHW
        features = self.backbone(input)
        # Apply positional encodings (if required)
        if self.positional_grid is not None:
            positional_encodings = self.positional_grid(
                features.shape[0], tuple(features.shape[-2:])
            )
            features = torch.cat([features, positional_encodings], dim=1)
        # Tokenize
        features = eo.rearrange(features, "b c h w -> b (h w) c")
        return features

    @property
    def output_dim(self):
        return self.backbone.dims[-1] + (
            self.positional_grid.dim if self.positional_grid is not None else 0
        )


class OutputTokenizer(nn.Module):
    pass


class FirstSlotAsLogits(OutputTokenizer):
    def forward(self, states: torch.Tensor):
        if states.dim() == 3:
            # states.shape = BUC
            return states[:, 0, :]
        else:
            assert states.dim() == 2
            return states


class OutputLatentPooling(OutputTokenizer):
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        softmax_type: str = "gumbel",
        softmax_temperature: float = 1.0,
    ):
        super(OutputLatentPooling, self).__init__()
        # Attries
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.softmax_type = softmax_type
        self.softmax_temperature = softmax_temperature
        # Modules
        self.confidence_scorer = nn.Linear(state_dim, 1)
        self.to_logits = nn.Linear(state_dim, output_dim)

    def softmax(self, confidences: torch.Tensor):
        # confidences.shape = BU
        if self.softmax_type == "gumbel":
            confidences = gumbel_softmax(
                confidences,
                tau=self.softmax_temperature,
                hard=False,
                dim=-1,
            )
        elif self.softmax_type == "gumbel-hard":
            confidences = gumbel_softmax(
                confidences,
                tau=self.softmax_temperature,
                hard=True,
                dim=-1,
            )
        elif self.softmax_type == "vanilla":
            confidences = torch.softmax(confidences / self.softmax_temperature, dim=-1)
        else:
            raise NotImplementedError(f"Unknown softmax type: {self.softmax_type}")
        return confidences

    def forward(self, states: torch.Tensor):
        # states.shape = BUC
        # confidence.shape = BU
        confidences = self.confidence_scorer(states)[..., 0]
        # After the softmax, confidences is normalized along all the output latents.
        confidences = self.softmax(confidences)
        # Weight each state with its respective confidence
        # selected_states.shape = BC
        selected_states = torch.einsum("buc,bu->bc", states, confidences)
        # With the state selected, time to compute the logits
        logits = self.to_logits(selected_states)
        return logits
