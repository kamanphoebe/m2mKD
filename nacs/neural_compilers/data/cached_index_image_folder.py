from typing import Optional, Callable, Any, Tuple

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


class CachedIndexImageFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        cache_path: str,
        transform: Optional[Callable],
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ):
        """
        Like ImageFolder, but doesn't crawl the image directory looking for files.
        Instead, relies on a cache.
        """
        super(CachedIndexImageFolder, self).__init__(
            root=root, transform=transform, target_transform=target_transform
        )
        self.loader = loader
        self._load_cache(cache_path)

    def _load_cache(self, path: str) -> None:
        cache = torch.load(path)
        self.classes = cache["classes"]
        self.class_to_idx = cache["class_to_idx"]
        self.samples = cache["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
