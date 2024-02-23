from typing import List, Tuple, Optional

import torch
import string

from torch.utils.data import IterableDataset


class Functions(IterableDataset):
    def __init__(
        self,
        fn: str,
        domain: List[Tuple[float, float]],
        batch_size: Optional[int] = None,
    ):
        self.fn = fn
        self.domain = domain
        self.batch_size = batch_size

    def sample(
        self, num_samples: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        num_samples = num_samples or self.batch_size
        assert num_samples is not None
        evaluation_points = [
            torch.empty(num_samples, 1).uniform_(a, b) for (a, b) in self.domain
        ]
        fn = (
            f"lambda "
            f"{', '.join(list(string.ascii_lowercase[0:len(evaluation_points)]))}: "
            f"{self.fn}"
        )
        samples = eval(fn)(*evaluation_points)
        return evaluation_points, samples

    def __iter__(self):
        while True:
            yield self.sample()
