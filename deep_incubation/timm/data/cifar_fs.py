import numpy as np
from typing import Callable, Optional
from torchvision.datasets import CIFAR100


class CIFAR100FS(CIFAR100):
    def __init__(
        self, 
        root: str, 
        chosen_class: list,
        n_sample: int,
        seed: int = None,
        train: bool = True, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        
        super(CIFAR100FS, self).__init__(root, transform=transform, target_transform=target_transform)
        
        if seed:
            np.random.seed(seed)
        chosen_idxs = np.array([], dtype=int)
        np_targets = np.array(self.targets)
        for cls in chosen_class:
            idxs = np.where(np_targets == cls)[0]
            n_idxs = np.random.permutation(idxs)[0:n_sample]
            chosen_idxs = np.concatenate((chosen_idxs, n_idxs))
        permute_idxs = np.random.permutation(chosen_idxs)
        self.data = self.data[permute_idxs]
        self.targets = np_targets[permute_idxs]
        # Map chosen classes to sequential numbers.
        chosen_class.sort()
        for idx, cls in enumerate(chosen_class):
            self.targets = np.where(self.targets == cls, idx, self.targets)
        self.targets = self.targets.tolist()
            
# dataset = CIFAR100FS('/share/project/liangyiming/datasets', [5, 20, 67], 3)