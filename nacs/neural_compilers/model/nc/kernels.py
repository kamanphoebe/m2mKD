import math
from contextlib import contextmanager
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli
#from torch_db import Tracer

from neural_compilers.utils import straight_through_clamp


def get_kernel(name):
    return globals()[name]


class Kernel(nn.Module):
    TRUNCATION_IMPLEMENTATIONS = {"direct", "masked"}

    def __init__(
        self,
        truncation: Optional[float] = None,
        straight_through: bool = True,
        initial_bandwidth: float = 1.0,
        learnable_bandwidth: bool = True,
        use_gumbel_along_functions: bool = False,
        gumbel_temperature: float = 1.0,
        gumbel_hard: bool = False,
        stochastic_kernel: bool = False,
        stochastic_sampling_temperature: float = 1.0,
        stochastic_eval_behaviour: str = "no_sample",
        truncation_implementation: str = "direct",
    ):
        super(Kernel, self).__init__()
        # Validate
        assert truncation_implementation in self.TRUNCATION_IMPLEMENTATIONS, (
            f"Unknown implementation: {truncation_implementation}; "
            f"it should be one of: {self.TRUNCATION_IMPLEMENTATIONS}."
        )
        # Privates
        self._return_distance = False
        self._return_log_kernel = False
        self._sample_kernel_if_required = True
        # Publics
        self.truncation = truncation
        self.straight_through = straight_through
        self.learnable_bandwidth = learnable_bandwidth
        self.initial_bandwidth = initial_bandwidth
        self.use_gumbel_along_functions = use_gumbel_along_functions
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_hard = gumbel_hard
        self.stochastic_kernel = stochastic_kernel
        self.stochastic_sampling_temperature = stochastic_sampling_temperature
        self.stochastic_eval_behaviour = stochastic_eval_behaviour
        self.truncation_implementation = truncation_implementation
        if self.learnable_bandwidth:
            self.bandwidth = nn.Parameter(
                torch.tensor(float(initial_bandwidth), dtype=torch.float)
            )
        else:
            self.register_buffer(
                "bandwidth", torch.tensor(float(initial_bandwidth), dtype=torch.float)
            )

    @contextmanager
    def return_distance(self):
        old_return_distance = self._return_distance
        self._return_distance = True
        yield
        self._return_distance = old_return_distance

    @contextmanager
    def return_log_kernel(self):
        old_return_log_kernel = self._return_log_kernel
        self._return_log_kernel = True
        yield
        self._return_log_kernel = old_return_log_kernel

    @contextmanager
    def do_not_sample_kernel(self):
        old_sample_kernel_if_required = self._sample_kernel_if_required
        self._sample_kernel_if_required = False
        yield
        self._sample_kernel_if_required = old_sample_kernel_if_required

    @property
    def returning_distance(self):
        return self._return_distance

    @property
    def returning_log_kernel(self):
        return self._return_log_kernel

    @property
    def sampling_kernel_if_required(self):
        return self._sample_kernel_if_required

    def gumbel_softmax_along_functions(self, distances: torch.Tensor) -> torch.Tensor:
        # Signature: B...UV -> B...UV
        # distances.shape = B...UV
        # We apply gumbel softmax along U.
        softmaxed_negative_distances = F.gumbel_softmax(
            -distances, dim=-2, tau=self.gumbel_temperature, hard=self.gumbel_hard
        )
        return softmaxed_negative_distances

    def truncated_gaussian(self, distance: torch.Tensor) -> torch.Tensor:
        # Early exit if we're returning distance
        if self.returning_distance:
            return distance
        kwargs = dict(
            distance=distance,
            bandwidth=self.bandwidth,
            truncation=self.truncation,
            straight_through=self.straight_through,
            returning_log_kernel=self.returning_log_kernel,
            use_gumbel_along_functions=self.use_gumbel_along_functions,
            gumbel_fn=self.gumbel_softmax_along_functions,
        )
        if self.truncation_implementation == "direct":
            kernel = self._direct_truncate(**kwargs)
        elif self.truncation_implementation == "masked":
            kernel = self._masked_truncate(**kwargs)
        else:
            raise NotImplementedError
        return kernel

    def sample_kernel(
        self, kernel: torch.Tensor, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        # If we're returning the distance or if no stochasticity is required,
        # this function is a no-op (id).
        if self.returning_distance or not self.stochastic_kernel:
            return kernel
        if self.training or self.stochastic_eval_behaviour == "sample":
            if self.returning_log_kernel:
                # Kernel is the log of a value between 0 and 1, meaning we'll first need
                # to exponentiate it. The resulting kernel gives the probs, which we
                # sample from, and convert back to log-space.
                # We need a straight_through_clamp to correct for floating point
                # errors earlier in the graph.
                proba_kernel = (
                    RelaxedBernoulli(
                        probs=straight_through_clamp(kernel.exp(), 0.0, 1.0),
                        temperature=self.stochastic_sampling_temperature,
                    )
                    .rsample(torch.Size() if batch_size is None else (batch_size,))
                    .log()
                )
            else:
                # Kernel should be a value between 0 and 1, meaning it can be interpreted as
                # a probability.
                # We need a straight_through_clamp to correct for floating point
                # errors earlier in the graph.
                proba_kernel = RelaxedBernoulli(
                    probs=straight_through_clamp(kernel.exp(), 0.0, 1.0),
                    temperature=self.stochastic_sampling_temperature,
                ).rsample(torch.Size() if batch_size is None else (batch_size,))
        elif self.stochastic_eval_behaviour == "threshold":
            if self.returning_log_kernel:
                proba_kernel = torch.where(kernel < math.log(0.5), -float("inf"), 0.0)
            else:
                proba_kernel = kernel.clone().gt_(0.5)
        elif self.stochastic_eval_behaviour == "no_sample":
            # If we're returning log kernel, the kernel is already in log-space.
            # If not, we're also in the right space, so nothing to do here.
            proba_kernel = kernel
        else:
            raise NotImplementedError
        return proba_kernel

    @staticmethod
    def _direct_truncate(
        distance: torch.Tensor,
        bandwidth: torch.Tensor,
        truncation: float,
        straight_through: bool = True,
        returning_log_kernel: bool = False,
        use_gumbel_along_functions: bool = False,
        gumbel_fn: Callable = None,
    ) -> torch.Tensor:
        # Scale distance by bandwidth right away for stable gradients
        distance = pre_truncation_distance = distance / bandwidth
        # Truncate distances if required. All distances above a threshold are
        # yeeted to inf.
        if truncation is not None:
            truncated_distance = torch.where(
                distance > (truncation / bandwidth.detach()),
                torch.empty_like(distance).fill_(float("inf")),
                distance,
            )
            if straight_through:
                distance = distance + (truncated_distance - distance).detach()
            else:
                distance = truncated_distance
        if returning_log_kernel:
            # In this case, we're returning the log of what would otherwise be the
            # output of this class.
            kernel = -distance
        elif use_gumbel_along_functions:
            assert gumbel_fn is not None
            pre_kernel = gumbel_fn(pre_truncation_distance)
            kernel = pre_kernel * torch.exp(-distance)
        else:
            # Exponentiate. This will have the min possible distance (= 0) always
            # mapped to 1.
            kernel = (-distance).exp()
        return kernel

    @staticmethod
    def _masked_truncate(
        distance: torch.Tensor,
        bandwidth: torch.Tensor,
        truncation: float,
        straight_through: bool = True,
        returning_log_kernel: bool = False,
        use_gumbel_along_functions: bool = False,
        gumbel_fn: Callable = None,  # noqa
    ) -> torch.Tensor:
        if truncation is not None:
            with torch.no_grad():
                mask = distance.clone().lt_(truncation)
        else:
            mask = None
        distance = distance / bandwidth
        if returning_log_kernel:
            if mask is not None:
                # Multiplying in exp space is adding in log space
                kernel = -distance + torch.log(mask)
            else:
                kernel = -distance
        elif use_gumbel_along_functions:
            raise NotImplementedError(
                "This implementation does not support "
                "use_gumbel_along_functions. Try the 'direct' implementation."
            )
        else:
            if mask is None:
                kernel = torch.exp(-distance)
            else:
                pre_kernel = torch.exp(-distance)
                # kernel is masked
                kernel = pre_kernel * mask
                if straight_through:
                    # The gradients go through pre_kernel, but the forward pass is
                    # through kernel.
                    kernel = pre_kernel + (kernel - pre_kernel).detach()
        return kernel

    def normalize_kernel_along_dim(
        self, kernel: torch.Tensor, dim: int
    ) -> torch.Tensor:
        # There are two cases of interest: we're returning log kernel or we're not.
        if self.returning_log_kernel:
            # kernel is in log-space, but we still want to normalize in exp space.
            kernel = kernel - torch.logsumexp(kernel, dim=dim, keepdim=True)
        else:
            # kernel is already in exp space, so not much to do here but norm
            kernel = kernel / torch.sum(kernel, dim=dim, keepdim=True)
        return kernel

    def forward(self, signatures: torch.Tensor, types: torch.Tensor):
        raise NotImplementedError


#@Tracer.register_tracer_probes("distance", "post_gaussian", "post_sampling")
class DotProductKernel(Kernel):
    def normalize(self, signatures: torch.Tensor, types: torch.Tensor):
        # signatures.shape = ...C
        # types.shape = ...C
        signatures = F.normalize(signatures, p=2, dim=-1)
        types = F.normalize(types, p=2, dim=-1)
        return signatures, types

    def forward(
        self,
        signatures: torch.Tensor,
        types: torch.Tensor,
        einsum_program: Optional[str] = None,
        normalize_kernel_along_dim: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        # Function signature: {BUC,UC}, {BVC, VC} -> {BUV, UV}
        # signatures.shape = UC or BUC
        # types.shape = VC or BVC
        # First normalize both variables before computing inner product
        signatures, types = self.normalize(signatures, types)
        # Compute the dot product distance,
        # defined as (1 - cosine_similarity)
        if einsum_program is not None:
            pass
        else:
            einsum_programs = {
                (2, 3): "uc,bvc->buv",
                (3, 3): "buc,bvc->buv",
                (2, 2): "uc,vc->uv",
            }
            signatures_dim = signatures.dim()
            types_dim = types.dim()
            einsum_program = einsum_programs[(signatures_dim, types_dim)]
        # distance.shape = BUV or UV
        distance = 1.0 - torch.einsum(einsum_program, signatures, types)
        #distance = Tracer.track(self, "distance", distance)
        # Compute and return the kernel
        kernel = self.truncated_gaussian(distance)
        #kernel = Tracer.track(self, "post_gaussian", kernel)
        # Sample if required
        if self.stochastic_kernel and self.sampling_kernel_if_required:
            assert not self.returning_distance, "Can't sample if returning distance!"
            kernel = self.sample_kernel(kernel, batch_size=batch_size)
        #kernel = Tracer.track(self, "post_sampling", kernel)
        # Normalize the kernel if required
        if normalize_kernel_along_dim is not None:
            kernel = self.normalize_kernel_along_dim(
                kernel, dim=normalize_kernel_along_dim
            )
        return kernel
