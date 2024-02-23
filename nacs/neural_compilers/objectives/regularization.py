from typing import Optional, Mapping, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Categorical
import einops as eo

from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from neural_compilers.utils import reduce_tensor, ScheduledHyperParameter


class WeightedSumLoss(nn.Module):
    def __init__(self, simplified_interface: bool = False):
        super(WeightedSumLoss, self).__init__()
        self.losses = nn.ModuleDict(dict())
        self.weights = nn.ModuleDict(dict())
        self.simplified_interface = simplified_interface

    def add_loss(
        self,
        loss: nn.Module,
        weight: Union[float, str, ScheduledHyperParameter],
        name: Optional[str] = None,
    ):
        assert name != "loss", "`loss` is a reserved name, please use something else."
        if name is None:
            name = f"loss_{len(self.losses)}"
            assert name not in self.losses
        self.losses[name] = loss
        if isinstance(weight, (float, int)):
            weight = ScheduledHyperParameter(float(weight))
        elif isinstance(weight, str):
            weight = ScheduledHyperParameter(1.0, schedule_spec=weight).update()
        else:
            assert isinstance(weight, ScheduledHyperParameter)
        self.weights[name] = weight
        return self

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss_terms = []
        unweighted_losses = {}
        for name, loss in self.losses.items():
            loss_value = loss(output, target)
            unweighted_losses[name] = loss_value
            loss_terms.append(loss_value * self.weights[name]())
        net_loss = torch.stack(loss_terms).sum()
        if self.simplified_interface:
            return net_loss
        else:
            assert "loss" not in unweighted_losses
            unweighted_losses["loss"] = net_loss
            return unweighted_losses


class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: nn.Module, teacher_model: nn.Module,
                 distillation_type: str, alpha: float = None, tau: float = None):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        # outputs_kd = None
        # if not isinstance(outputs, torch.Tensor):
        #     # assume that the model outputs a tuple of [outputs, outputs_kd]
        #     outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss
        
        loss_dict = {}
        base_loss = base_loss['loss']
        loss_dict['base_loss'] = base_loss
        outputs_kd = outputs
        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        
        loss_dict['kd_loss'] = distillation_loss
        loss_dict['loss'] = loss
        return loss_dict


class SignatureHingeRepulsion(nn.Module):
    def __init__(self, model: nn.Module, hinge: float, reduction: str = "sum"):
        super(SignatureHingeRepulsion, self).__init__()
        self.model = model
        self.hinge = hinge
        self.reduction = reduction

    # noinspection PyUnusedLocal
    def forward(
        self,
        output: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def extract_signatures(latent_graph):
            input_signatures = latent_graph.input_latents.signatures
            mediator_signatures = latent_graph.mediator_latents.signatures
            output_signatures = latent_graph.output_latents.signatures
            return torch.cat(
                [input_signatures, mediator_signatures, output_signatures], dim=0
            )

        all_signatures = [
            extract_signatures(latent_graph)
            for latent_graph in self.model.latent_graphs
        ]
        all_kernels = [
            latent_graph.propagators[0].latent_attention.kernel
            for latent_graph in self.model.latent_graphs
        ]
        loss_terms = []
        tril_mask = None
        for signatures, kernel in zip(all_signatures, all_kernels):
            with kernel.return_distance():
                with kernel.do_not_sample_kernel():
                    # distance_matrix.shape = UU
                    distance_matrix = kernel(signatures, signatures)
            # Make the tril_mask if it's not already made
            if (tril_mask is None) or (tril_mask.shape != distance_matrix.shape):
                with torch.no_grad():
                    tril_mask = torch.tril(
                        torch.ones_like(distance_matrix, dtype=torch.bool), diagonal=-1
                    )
            # distances.shape = N
            distances = distance_matrix[tril_mask]
            # Select all distances that are below the threshold and whip them
            with torch.no_grad():
                too_close = distances.lt(self.hinge)
            distances = distances[too_close]
            loss_terms.append(-reduce_tensor(distances, mode=self.reduction, dim=0))
        loss = torch.stack(loss_terms).sum()
        return loss


class SignatureDistributionRegularization(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_components: int = 1,
        reduction: str = "mean",
        reduction_along_graphs: str = "sum",
    ):
        super(SignatureDistributionRegularization, self).__init__()
        self.model = model
        self.num_components = num_components
        self.reduction = reduction
        self.reduction_along_graphs = reduction_along_graphs
        self._compute_gaussian_components()

    @staticmethod
    def extract_signatures(latent_graph):
        input_signatures = latent_graph.input_latents.signatures
        mediator_signatures = latent_graph.mediator_latents.signatures
        output_signatures = latent_graph.output_latents.signatures
        return torch.cat(
            [input_signatures, mediator_signatures, output_signatures], dim=0
        )

    def _get_distances(self):
        ret_vals = []
        for latent_graph in self.model.latent_graphs:
            signatures = self.extract_signatures(latent_graph)
            kernel = latent_graph.propagators[0].latent_attention.kernel
            with kernel.return_distance():
                with kernel.do_not_sample_kernel():
                    distance_matrix = kernel(signatures, signatures)
            with torch.no_grad():
                tril_mask = torch.tril(
                    torch.ones_like(distance_matrix, dtype=torch.bool), diagonal=-1
                )
            distances = distance_matrix[tril_mask]
            ret_vals.append(
                dict(
                    signatures=signatures,
                    distances=distances,
                )
            )
        return ret_vals

    def _compute_gaussian_components(self):
        distances = [d["distances"] for d in self._get_distances()]
        mixture_weights = []
        mixture_means = []
        mixture_scales = []
        for _distances in distances:
            gmm = GaussianMixture(
                n_components=self.num_components, covariance_type="spherical"
            )
            gmm.fit(_distances.data.cpu().numpy()[:, None])
            mixture_means.append(torch.from_numpy(gmm.means_[:, 0]).float())
            mixture_scales.append(torch.from_numpy(gmm.covariances_).float().sqrt_())
            mixture_weights.append(torch.from_numpy(gmm.weights_).float())
        # component_{weights, means, sigmas}.shape = (num_layers, num_components)
        self.register_buffer("component_weights", torch.stack(mixture_weights))
        self.register_buffer("component_means", torch.stack(mixture_means))
        self.register_buffer("component_sigmas", torch.stack(mixture_scales))

    def forward(
        self,
        output: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distances = torch.stack([d["distances"] for d in self._get_distances()], dim=0)
        log_probs = self._compute_log_probs(distances)
        loss = reduce_tensor(
            reduce_tensor(-log_probs, mode=self.reduction, dim=1),
            mode=self.reduction_along_graphs,
            dim=0,
        )
        return loss

    def _compute_log_probs(self, distances: torch.Tensor) -> torch.Tensor:
        # distances.shape = (num_layers, num_pairs)
        component_distribution = Normal(
            loc=self.component_means, scale=self.component_sigmas
        )
        weight_distribution = Categorical(probs=self.component_weights)
        # component_log_prob.shape = (num_pairs, num_layers, num_components)
        component_log_prob = component_distribution.log_prob(
            eo.rearrange(distances, "layers pairs -> pairs layers ()")
        )
        # mixture_weight_log_prob.shape = (num_layers, num_components)
        mixture_weight_log_prob = torch.log_softmax(weight_distribution.logits, dim=-1)
        log_prob = eo.rearrange(
            torch.logsumexp(component_log_prob + mixture_weight_log_prob[None], dim=-1),
            "pairs layers -> layers pairs",
        )
        return log_prob


class StochasticBlockModelRegularization(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_clusters: int = 10,
        p_in: float = 0.9,
        p_out: float = 0.05,
        regularize_input_latents: bool = True,
        regularize_mediator_latents: bool = True,
        regularize_output_latents: bool = True,
        clip: Optional[float] = None,
    ):
        super(StochasticBlockModelRegularization, self).__init__()
        # Only supported for a single latent_graph for now
        assert (
            len(model.latent_graphs) == 1
        ), "Only support for a single latent graph for now."
        self.model = model
        self.num_clusters = num_clusters
        self.p_in = p_in
        self.p_out = p_out
        self.regularize_input_latents = regularize_input_latents
        self.regularize_mediator_latents = regularize_mediator_latents
        self.regularize_output_latents = regularize_output_latents
        self.clip = clip
        # Get the graphon and register it as buffer
        self.register_buffer("sampled_graphon", self.get_sampled_graphon())

    @staticmethod
    def match_and_permute(input_adjacency: torch.Tensor, target_graphon: torch.Tensor):
        # input_adjacency is assumed to be a matrix of probabilities, i.e.
        # containing values between 0 and 1.
        # input_adjacency.shape = UV
        # target_graphon.shape = UV
        # Construct cost matrix based on MSE between pairs of rows
        # cost_matrix.shape = UU
        with torch.no_grad():
            cost_matrix = (
                (target_graphon[:, None, :] - input_adjacency[None, :, :])
                .pow(2)
                .mean(-1)
            )
            target_idx, input_idx = linear_sum_assignment(
                cost_matrix=cost_matrix.detach().cpu().numpy(), maximize=False
            )
            # Permute the inputs
            input_idx = torch.from_numpy(input_idx).long().to(input_adjacency.device)
        input_adjacency = input_adjacency[input_idx]
        return input_adjacency

    @property
    def num_regularized_latents(self) -> int:
        num = 0
        if self.regularize_input_latents:
            num += self.model.latent_graphs[0].input_latents.num_latents
        if self.regularize_mediator_latents:
            num += self.model.latent_graphs[0].mediator_latents.num_latents
        if self.regularize_output_latents:
            num += self.model.latent_graphs[0].output_latents.num_latents
        return num

    @property
    def cluster_size(self) -> int:
        assert self.num_regularized_latents % self.num_clusters == 0
        return self.num_regularized_latents // self.num_clusters

    def get_sampled_graphon(self) -> torch.Tensor:
        sampled_graphon = torch.zeros(
            self.num_regularized_latents, self.num_regularized_latents
        ).add_(self.p_out)
        num_clusters = self.num_clusters
        cluster_size = self.cluster_size
        for cluster_idx in range(num_clusters):
            sl = slice(cluster_idx * cluster_size, (cluster_idx + 1) * cluster_size)
            sampled_graphon[sl, sl] = self.p_in
        return sampled_graphon

    @property
    def kernel_is_unique(self) -> bool:
        kernel_hashes = [
            (
                propagator.latent_attention.kernel.initial_bandwidth,
                propagator.latent_attention.kernel.truncation,
            )
            for propagator in self.model.latent_graphs[0].propagators
        ]
        kernel_learnable = any(
            [
                propagator.latent_attention.kernel.learnable_bandwidth
                for propagator in self.model.latent_graphs[0].propagators
            ]
        )
        return not kernel_learnable and all(
            [kernel_hash == kernel_hashes[0] for kernel_hash in kernel_hashes]
        )

    def _fetch_signatures(self) -> torch.Tensor:
        signatures = []
        if self.regularize_input_latents:
            signatures.append(self.model.latent_graphs[0].input_latents.signatures)
        if self.regularize_mediator_latents:
            signatures.append(self.model.latent_graphs[0].mediator_latents.signatures)
        if self.regularize_output_latents:
            signatures.append(self.model.latent_graphs[0].output_latents.signatures)
        return torch.cat(signatures, dim=0)

    def _get_link_probabilities(self) -> List[torch.Tensor]:
        # If we're learning the bandwidths, each propagator might learn a different one.
        # But if we're not, it will be wasted compute if we evaluate the matching multiple
        # times.
        kernels = []
        if self.kernel_is_unique:
            kernels.append(
                self.model.latent_graphs[0].propagators[0].latent_attention.kernel
            )
        else:
            for propagator in self.model.latent_graphs[0].propagators:
                kernels.append(propagator.latent_attention.kernel)
        # Get the sigs
        signatures = self._fetch_signatures()
        # Init buffer
        link_probas = []
        for kernel in kernels:
            # Compute the link probas
            with kernel.do_not_sample_kernel():
                link_proba = kernel(signatures, signatures)
            # link_proba.shape = UV
            assert link_proba.dim() == 2
            link_probas.append(link_proba)
        # Done
        return link_probas

    def _compute_matched_loss(self, link_probability: torch.Tensor):
        if self.clip is None or self.clip == 0.0:
            # no-clip loss is just the simple MSE
            return F.mse_loss(link_probability, self.sampled_graphon)
        else:
            # We kill loss terms below the threshold specified by clip
            # unreduced_loss.shape = UV
            unreduced_loss = F.mse_loss(
                link_probability, self.sampled_graphon, reduction="none"
            )
            return torch.where(
                unreduced_loss < self.clip,
                torch.zeros_like(unreduced_loss).add_(self.clip),
                unreduced_loss,
            ).mean()

    def forward(
        self,
        output: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get the link probabilities; these are the adjacency matrices, one for each
        # unique kernel in the model.
        link_probabilities = self._get_link_probabilities()
        # Buffer for storing losses
        losses = []
        for link_probability in link_probabilities:
            # Permute link_probabilities based on the target graphon
            link_probability = self.match_and_permute(
                link_probability, self.sampled_graphon
            )
            # Compute the MSE
            loss = self._compute_matched_loss(link_probability)
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss
