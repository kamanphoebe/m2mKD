from typing import Optional, Callable

import yaml
import asset

import numpy as np
from torchvision.datasets import ImageFolder


WNIDS = yaml.load(
    asset.load("neural_compilers:data/assets/wnids.yml").read(), Loader=yaml.SafeLoader
)


class ImageNetRTargetTransform:
    def __init__(self):
        inet_r_in_inet_mask = np.array(
            [(wnid in WNIDS["imagenet_r"]) for wnid in WNIDS["imagenet"]]
        )
        inet_indices = np.array(list(range(len(WNIDS["imagenet"]))))
        self.inet_r_to_inet_indices = inet_indices[inet_r_in_inet_mask]

    def __call__(self, target):
        return self.inet_r_to_inet_indices[target]


class ImageNet2ImageNetRPredictionTransform:
    def __init__(self):
        self.inet_r_in_inet_mask = [
            (wnid in WNIDS["imagenet_r"]) for wnid in WNIDS["imagenet"]
        ]

    def __call__(self, prediction):
        return prediction[..., self.inet_r_in_inet_mask]


class ImageNetR(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, **super_kwargs):
        super(ImageNetR, self).__init__(
            root,
            transform=transform,
            **super_kwargs,
        )
        self.prediction_transform = ImageNet2ImageNetRPredictionTransform()


class TinyImageNetRTargetTransform:
    def __init__(self):
        inet_r_in_inet_mask = np.array(
            [(wnid in WNIDS["tiny-imagenet-r"]) for wnid in WNIDS["tiny-imagenet-200"]]
        )
        inet_indices = np.array(list(range(len(WNIDS["tiny-imagenet-200"]))))
        self.inet_r_to_inet_indices = inet_indices[inet_r_in_inet_mask]

    def __call__(self, target):
        return self.inet_r_to_inet_indices[target]


class TinyImageNet2TinyImageNetRPredictionTransform:
    def __init__(self):
        self.inet_r_in_inet_mask = [
            (wnid in WNIDS["tiny-imagenet-r"]) for wnid in WNIDS["tiny-imagenet-200"]
        ]

    def __call__(self, prediction):
        return prediction[..., self.inet_r_in_inet_mask]


class TinyImageNetR(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, **super_kwargs):
        super(TinyImageNetR, self).__init__(
            root,
            transform=transform,
            **super_kwargs,
        )
        self.prediction_transform = TinyImageNet2TinyImageNetRPredictionTransform()
