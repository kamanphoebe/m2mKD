import math
import numpy as np
import networkx as nx
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as eo

from neural_compilers.model.nc.utils import VectorScaleNoise
from neural_compilers.utils import override


class Latents(nn.ParameterDict):
    def __init__(
        self,
        num_latents: int,
        signature_dim: int,
        code_dim: int,
        state_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        graph_preset: Optional[str] = None,
        graph_generator_kwargs: Optional[dict] = None,
        learnable_signatures: bool = True,
        learnable_codes: bool = True,
        init_with_identical_codes: bool = False,
        learnable_initial_states: bool = False,
        predefined_codes_path: Optional[str] = None,
    ):
        signatures = nn.Parameter(
            self._initialize_signatures(
                num_latents, signature_dim, graph_preset, graph_generator_kwargs
            )
        )
        if not learnable_signatures:
            signatures.requires_grad = False
        # We'll need a special code-path for num_layers = None, because otherwise loading
        # in previous checkpoints will be borky business. This justifies the mess.
        if init_with_identical_codes:
            if num_layers is None:
                shape = (code_dim,)
                repeat_key = "c -> n c"
            else:
                shape = (num_layers, code_dim)
                repeat_key = "l c -> l n c"
            if predefined_codes_path:
                codes = nn.Parameter(torch.load(predefined_codes_path))
                codes.requires_grad = True
                assert codes.shape == shape
            else:
                codes = nn.Parameter(
                    eo.repeat(torch.randn(*shape), repeat_key, n=num_latents)
                )
        else:
            if num_layers is None:
                shape = (num_latents, code_dim)
            else:
                shape = (num_layers, num_latents, code_dim)
            if predefined_codes_path:
                codes = nn.Parameter(torch.load(predefined_codes_path))
                codes.requires_grad = True
                assert codes.shape == shape
            else:
                codes = nn.Parameter(torch.randn(*shape))
        if not learnable_codes:
            codes.requires_grad = False
        if learnable_initial_states:
            assert state_dim is not None
            # Defined only if needed, in order to not have dangling parameters
            initial_states_dict = dict(
                initial_states_=nn.Parameter(torch.randn(num_latents, state_dim))
            )
        else:
            initial_states_dict = dict()
        super(Latents, self).__init__(
            dict(signatures_=signatures, codes_=codes, **initial_states_dict)
        )

    @staticmethod
    def _initialize_signatures(
        num_latents: int,
        signature_dim: int,
        graph_preset: Optional[str] = None,
        graph_generator_kwargs: Optional[dict] = None,
    ):

        if graph_preset is None:
            return torch.randn(num_latents, signature_dim)
        elif graph_preset == "complete":
            return eo.repeat(
                torch.tensor([1.0] + [0.0] * (signature_dim - 1)),
                "c -> n c",
                n=num_latents,
            )
        else:
            return preset_graph_signatures(
                num_latents=num_latents,
                signature_dim=signature_dim,
                graph_type=graph_preset,
                **(graph_generator_kwargs or {}),
            )

    @property
    def num_latents(self):
        return self["signatures_"].shape[0]

    @property
    def signature_dim(self):
        return self["signatures_"].shape[-1]

    @property
    def code_dim(self):
        return self["codes_"].shape[-1]

    @property
    def defines_initial_states(self):
        return "initial_states_" in self

    @property
    def initial_state_dim(self):
        if self.defines_initial_states:
            return self["initial_states_"].shape[-1]
        else:
            return 0

    @property
    def signatures(self):
        return self["signatures_"]

    @property
    def codes(self):
        return self["codes_"]

    @property
    def num_layers(self):
        if self["codes_"].dim() == 3:
            return self["codes_"].shape[0]
        else:
            return None

    def get_code(self, layer_idx: Optional[int] = None):
        if self.num_layers is None:
            # All layers have the same code
            return self.codes
        else:
            assert layer_idx is not None
            return self["codes_"][layer_idx]

    @property
    def initial_states(self):
        if self.defines_initial_states:
            return self["initial_states_"]
        else:
            return None

    def get_initial_states(self, batch_size: int):
        initial_states = self.initial_states
        assert initial_states is not None, "Latent doesn't define initial states."
        return eo.repeat(initial_states, "n c -> b n c", b=batch_size)


class LatentSeeder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        code_dim: int,
        code_noise_scale: Optional[float] = None,
        use_state_noise: bool = True,
        use_code_as_mean_state: bool = False,
    ):
        super(LatentSeeder, self).__init__()
        if use_code_as_mean_state:
            assert state_dim == code_dim
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.use_state_noise = use_state_noise
        self.use_code_as_mean_state = use_code_as_mean_state
        # Modules
        if not use_code_as_mean_state:
            self.layernorm = nn.LayerNorm(self.code_dim)
            self.code_to_state_mu = nn.Linear(self.code_dim, self.state_dim)
        else:
            assert not use_state_noise
            self.layernorm = None
            self.code_to_state_mu = None
        if use_state_noise:
            self.code_to_state_logvar = nn.Linear(self.code_dim, self.state_dim)
        else:
            self.code_to_state_logvar = None
        if code_noise_scale is not None:
            self.code_noiser = VectorScaleNoise(code_noise_scale)
        else:
            self.code_noiser = None
        self.init()

    def init(self):
        if self.code_to_state_logvar is not None:
            # Special init
            self.code_to_state_logvar.bias.data.fill_(-5.0)

    def forward(self, codes: torch.Tensor, batch_size: int):
        # codes.shape = UC
        if self.code_noiser is not None:
            codes = self.code_noiser(codes)
        if self.use_code_as_mean_state:
            state_mu = codes
        else:
            # Normalize
            codes = self.layernorm(codes)
            # state_{mu,std}.shape = UC
            state_mu = self.code_to_state_mu(codes)
        if self.use_state_noise:
            state_std = self.code_to_state_logvar(codes).mul(0.5).exp()
            random_tensor = torch.randn(
                batch_size,
                state_mu.shape[0],
                state_mu.shape[1],
                device=state_mu.device,
                dtype=state_mu.dtype,
            )
            # seed_state.shape = BUC
            seed_state = state_mu + random_tensor * state_std
        else:
            seed_state = eo.repeat(state_mu, "u c -> b u c", b=batch_size)
        return seed_state


def create_graph(graph_generator_kwargs: dict, num_latents: int, graph_type: str):
    if graph_type == "watts-strogatz":
        graph = nx.watts_strogatz_graph(
            n=num_latents,
            **override(graph_generator_kwargs, k=round(0.1 * num_latents), p=0.4),
        )
    elif graph_type == "planted-partition":
        assert (
            graph_generator_kwargs.get("l") * graph_generator_kwargs.get("k")
            == num_latents
        )
        graph = nx.planted_partition_graph(**graph_generator_kwargs)
    elif graph_type == "grid-graph":
        # graph_generator_kwargs = {"dim": (16,16), "periodic": False}
        assert (
            graph_generator_kwargs.get("dim")[0] * graph_generator_kwargs.get("dim")[1]
            == num_latents
        )
        graph = nx.grid_graph(**graph_generator_kwargs)
    elif graph_type == "balanced-tree":
        # graph_generator_kwargs = {"r": 2, "h": 7}
        assert (
            graph_generator_kwargs.get("r") ** graph_generator_kwargs.get("h") + 1
            == num_latents
        )
        graph = nx.generators.classic.balanced_tree(**graph_generator_kwargs)
        # need to add an additional edge to make it square
        last_node = graph.number_of_nodes() - 1
        new_node = last_node + 1
        graph.add_node(new_node)
        graph.add_edge(last_node, new_node)
    elif graph_type == "barbell":
        # graph_generator_kwargs = {"m1": 128, "m2": 0}
        # assert graph_generator_kwargs.get("m1") * 2 == num_latents
        graph = nx.barbell_graph(graph_generator_kwargs.get("m1"), 0)
    elif graph_type == "erdos-renyi":
        # graph_generator_kwargs = {"n": 256, "p": 0.1}
        assert graph_generator_kwargs.get("n") == num_latents
        graph = nx.generators.random_graphs.erdos_renyi_graph(**graph_generator_kwargs)
    elif graph_type == "barabasi-albert":
        # graph_generator_kwargs = {"n": 256, "m": 2}
        assert graph_generator_kwargs.get("n") == num_latents
        graph = nx.generators.random_graphs.barabasi_albert_graph(
            **graph_generator_kwargs
        )
    elif graph_type == "ring-of-cliques":
        # graph_generator_kwargs = {"num_cliques": 8, "clique_size": 32}
        assert (
            graph_generator_kwargs.get("num_cliques")
            * graph_generator_kwargs.get("clique_size")
            == num_latents
        )
        graph = nx.generators.community.ring_of_cliques(**graph_generator_kwargs)
    elif graph_type == "hypercube":
        # assert num_latents is a power of 2
        assert math.log(num_latents, 2).is_integer()
        graph = nx.hypercube_graph(8)
    else:
        raise NotImplementedError
    return graph


def preset_graph_signatures(
    num_latents: int,
    signature_dim: int,
    graph_type: str = "watts-strogatz",
    graph_generator_kwargs: Optional[dict] = None,
    truncation: float = 0.8,
    num_iters: int = 1000,
    learning_rate: float = 0.001,
    return_scores: bool = False,
    use_hinge_loss: bool = True,
    **graph_constructor_kwargs,
):
    # dev_env = initialize_device()
    graph = create_graph(
        {**graph_constructor_kwargs, **(graph_generator_kwargs or {})},
        num_latents,
        graph_type,
    )
    mse = torch.nn.MSELoss()

    def hinge_loss_fn(signatures, target):
        # signatures.shape = UC
        # Compute the cosine distance
        signatures = F.normalize(signatures, p=2, dim=-1)
        distances = 1.0 - torch.einsum("uc,vc->uv", signatures, signatures)
        # noinspection PyTypeChecker
        loss_value = torch.where(
            target == 1,
            torch.maximum(torch.ones_like(distances).mul_(truncation), distances),
            torch.maximum(torch.ones_like(distances).mul_(-truncation), -distances),
        ).mean()
        return loss_value

    def mse_loss_fn(signatures, target):
        # signatures.shape = UC
        signatures = F.normalize(signatures, p=2, dim=-1)
        distances = 1.0 - torch.einsum("uc,vc->uv", signatures, signatures)
        # noinspection PyTypeChecker
        loss_value = mse(distances, target)
        return loss_value

    @torch.no_grad()
    def hinge_score_fn(signatures, target):
        signatures = F.normalize(signatures, p=2, dim=-1)
        distances = 1.0 - torch.einsum("uc,vc->uv", signatures, signatures)
        predicted_adjacency = distances.lt_(truncation) - torch.eye(
            num_latents, dtype=torch.float
        )
        score = (predicted_adjacency - target).abs().mean()
        return score

    @torch.no_grad()
    def mse_score_fn(signatures, target):
        signatures = F.normalize(signatures, p=2, dim=-1)
        distances = 1.0 - torch.einsum("uc,vc->uv", signatures, signatures)
        score = mse(distances, target)
        return score

    # Now, initialize the signatures as orthogonal vectors
    learned_signatures = nn.init.orthogonal_(
        torch.empty(num_latents, signature_dim)
    ).requires_grad_()

    # Now, optimize
    optim = torch.optim.Adam([learned_signatures], lr=learning_rate)

    if use_hinge_loss:
        # The hinge loss ensures that the signatures are moved just enough, such that
        # when truncated at the specified truncation, the resulting graph matches the
        # target graph (except the self connections, which are always present).
        graph_target = torch.from_numpy(nx.convert_matrix.to_numpy_array(graph))
        loss_fn = hinge_loss_fn
        score_fn = hinge_score_fn
    else:
        # The MSE loss ensures that the distance between signatures of two nodes
        # approximates the normalized path distance between the said nodes. The
        # normalization is with respect to the maximum distances in the graph.
        path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
        target_distances = np.array(
            [
                [path_lengths.get(m, {}).get(n, 0) for m in graph.nodes]
                for n in graph.nodes
            ],
            dtype=np.float32,
        )
        target_distances = torch.tensor(target_distances, requires_grad=False)
        target_distances_scaled = target_distances / target_distances.max()
        graph_target = target_distances_scaled
        loss_fn = mse_loss_fn
        score_fn = mse_score_fn

    for iter_num in range(num_iters):
        loss = loss_fn(learned_signatures, graph_target)
        optim.zero_grad()
        loss.backward()
        optim.step()

    if not return_scores:
        return learned_signatures.data
    return learned_signatures.data, score_fn(learned_signatures, graph_target)
