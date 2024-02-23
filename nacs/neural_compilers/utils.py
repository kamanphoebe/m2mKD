import math
from copy import deepcopy
from functools import lru_cache
from typing import Union, Callable, Optional, List, Iterable, Mapping, Any, Dict
import inspect

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from einops import rearrange
from speedrun.utils.py_utils import flatten_dict, unflatten_dict, recursive_update


def make_mlp(
    in_features: int,
    out_features: int,
    capacity: int = None,
    num_layers: int = 2,
    activation: Union[Callable, None] = nn.GELU,
    context_channels: Optional[int] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
    trailing_dropout: bool = False,
    trailing_activation: bool = False,
    trailing_bias: bool = True,
    leading_dropout: bool = False,
    leading_activation: bool = False,
    leading_batch_norm: bool = False,
    mod_fc_kwargs: Optional[dict] = None,
) -> nn.Module:
    """Utility function for rolling a quick MLP."""
    # Infer types
    if context_channels is None:
        # No ModLins involved
        linear = nn.Linear
        sequential = nn.Sequential
    else:
        # ModLins involved
        linear = lambda in_, out_: ModFC(
            in_, out_, context_channels, **(mod_fc_kwargs or {})
        )  # noqa
        sequential = ModSequential
        activation_fn = deepcopy(activation)
        activation = (
            (lambda: StatelessWrapper(activation_fn()))
            if activation is not None
            else activation
        )

    # Make the model
    layer_sequence = []
    if batch_norm and leading_batch_norm:
        layer_sequence.append(nn.BatchNorm1d(in_features))
    if activation is not None and leading_activation:
        layer_sequence.append(activation())
    if dropout > 0.0 and leading_dropout:
        layer_sequence.append(nn.Dropout(p=dropout))
    if num_layers == 1:
        layer_sequence.append(linear(in_features, out_features, bias=trailing_bias))
    elif num_layers >= 2:
        assert activation is not None
        layer_sequence.append(linear(in_features, capacity))
        if batch_norm:
            layer_sequence.append(nn.BatchNorm1d(capacity))
        layer_sequence.append(activation())
        if dropout > 0.0:
            layer_sequence.append(nn.Dropout(p=dropout))
        for layer_num in range(num_layers - 2):
            layer_sequence.append(linear(capacity, capacity))
            if batch_norm:
                layer_sequence.append(nn.BatchNorm1d(capacity))
            layer_sequence.append(activation())
            if dropout > 0.0:
                layer_sequence.append(nn.Dropout(p=dropout))
        layer_sequence.append(linear(capacity, out_features, bias=trailing_bias))
    else:
        raise ValueError
    if activation is not None and trailing_activation:
        layer_sequence.append(activation())
    if dropout > 0.0 and trailing_dropout:
        layer_sequence.append(nn.Dropout(p=dropout))
    if len(layer_sequence) == 1:
        return layer_sequence[0]
    else:
        return sequential(*layer_sequence)


def get_activation(activation: Union[str, nn.Module, type, None]) -> nn.Module:
    if isinstance(activation, str):
        activation = getattr(torch.nn, activation)()
    elif isinstance(activation, nn.Module):
        pass
    elif isinstance(activation, type):
        activation = activation()
    else:
        assert activation is None, f"Can't parse: {activation}"
    return activation


def get_weight_init_fn(weight_init_fn: Union[str, Callable, None]) -> Callable:
    if isinstance(weight_init_fn, str):
        # Get the right fn from torch init
        if hasattr(torch.nn.init, weight_init_fn):
            weight_init_fn = getattr(torch.nn.init, weight_init_fn)
        else:
            assert weight_init_fn in globals()
            weight_init_fn = globals()[weight_init_fn]
    elif callable(weight_init_fn):
        pass
    else:
        assert weight_init_fn is None
        weight_init_fn = lambda p: nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    return weight_init_fn


def trunc_normal_switch_weight_init_(x, gain: float = 1):
    assert x.dim() == 2, "This will be borky if x is a conv filter, but meh for now"
    fan_out, fan_in = x.shape[0:2]
    nn.init.trunc_normal_(x, mean=0.0, std=np.sqrt(gain / fan_in))
    return x


class ModFC(nn.Module):
    VALID_SCALE_NORMALIZATION_TYPES = ["ln", "l2", "none", "sigmoid", "tanh"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        bias: bool = True,
        *,
        activation: Optional[Callable] = None,
        eps: float = 1e-6,
        scale_normalization_type: str = "ln",
        scale_gain: float = 1.0,
        learnable_scale_gain: bool = False,
        scale_dropout: float = 0.0,
        add_one_to_scale: bool = False,
        scale_times_input_normalization_type: str = "none",
        weight_init_fn: Union[str, Callable, None] = None,
        weight_init_fn_kwargs: Optional[dict] = None,
        weight_init_gain: float = 1.0,
    ):
        super(ModFC, self).__init__()
        # Validate args
        assert scale_normalization_type in self.VALID_SCALE_NORMALIZATION_TYPES
        assert (
            scale_times_input_normalization_type in self.VALID_SCALE_NORMALIZATION_TYPES
        )
        # Build params
        self.native_weight, self.native_bias = self.init_weights_and_bias(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            weight_init_fn=weight_init_fn,
            weight_init_fn_kwargs=weight_init_fn_kwargs,
            weight_init_gain=weight_init_gain,
        )
        self.scale = nn.Linear(context_features, in_features, bias=False)
        if scale_dropout > 0.0:
            self.scale_dropout = nn.Dropout(p=scale_dropout)
        else:
            assert scale_dropout == 0.0
            self.scale_dropout = nn.Identity()
        self.activation = get_activation(activation)
        self.eps = eps
        self.scale_normalization_type = scale_normalization_type
        if not learnable_scale_gain:
            self.scale_gain = scale_gain
        else:
            self.scale_gain = nn.Parameter(torch.tensor(scale_gain))
        self.scale_times_input_normalization_type = scale_times_input_normalization_type
        self.add_one_to_scale = add_one_to_scale

    @staticmethod
    def init_weights_and_bias(
        in_features: int,
        out_features: int,
        bias: bool,
        weight_init_fn: Callable = None,
        weight_init_fn_kwargs: Optional[dict] = None,
        weight_init_gain: float = 1.0,
    ):
        weight_init_fn = get_weight_init_fn(weight_init_fn)
        if weight_init_fn_kwargs is None:
            weight_init_fn_kwargs = dict()
        weights = nn.Parameter(
            weight_init_fn(
                torch.empty(out_features, in_features), **weight_init_fn_kwargs
            ).mul_(weight_init_gain)
        )
        if bias:
            bias = nn.Parameter(torch.zeros(out_features))
        else:
            bias = None
        return weights, bias

    def _naive_implementation(
        self, input: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        # input.shape = ...C
        # context.shape = ...C
        # scale.shape = ...C
        if self.scale_normalization_type == "ln":
            scale = self.scale_gain * F.layer_norm(
                self.scale(context), [input.shape[-1]]
            )
        elif self.scale_normalization_type == "l2":
            scale = self.scale_gain * F.normalize(self.scale(context), dim=-1, p=2)
        elif self.scale_normalization_type == "none":
            scale = self.scale_gain * self.scale(context)
        elif self.scale_normalization_type == "sigmoid":
            scale = self.scale_gain * torch.sigmoid(self.scale(context))
        elif self.scale_normalization_type == "tanh":
            scale = self.scale_gain * torch.tanh(self.scale(context))
        else:
            raise NotImplementedError
        # This is a noop if no scale dropout is requested
        scale = self.scale_dropout(scale)
        if self.add_one_to_scale:
            scale = scale + 1.0
        if self.scale_times_input_normalization_type == "none":
            input_times_scale = input * scale
        elif self.scale_times_input_normalization_type == "ln":
            input_times_scale = F.layer_norm(input * scale, [input.shape[-1]])
        elif self.scale_times_input_normalization_type == "l2":
            input_times_scale = F.normalize(input * scale, dim=-1, p=2)
        else:
            raise NotImplementedError
        output = F.linear(input_times_scale, self.native_weight, bias=self.native_bias)
        # Activate if required
        if self.activation is not None:
            output = self.activation(output)
        return output

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        return self._naive_implementation(input, context)


class FC(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        bias: bool = True,
        *,
        activation: Optional[Callable] = None,
    ):
        super(FC, self).__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        output = self.lin(input)
        if self.activation is not None:
            output = self.activation(output)
        return output


class WModFC(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        bias: bool = True,
        *,
        activation: Optional[Callable] = None,
        eps: float = 1e-6,
    ):
        super(WModFC, self).__init__()
        self.modulation = nn.Linear(context_features, in_features)
        self.weight = nn.Parameter(
            get_weight_init_fn(None)(torch.empty(out_features, in_features))
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.activation = get_activation(activation)
        self.eps = eps

    def forward(self, input: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # input.shape = BNI
        assert input.dim() >= 2
        assert context.dim() == 2
        assert input.shape[-2] == context.shape[-2]
        # context.shape = NI
        context = self.modulation(context) + 1.0
        # self.weight.shape = OI
        # weight.shape = NOI
        modulated_weight = torch.einsum("oi,ni->noi", self.weight, context)
        # normalized_weight.shape = NOI
        normalized_weight = modulated_weight * torch.rsqrt(
            modulated_weight.pow(2).sum(-1, keepdims=True) + self.eps
        )
        # output.shape = ...O
        output = torch.einsum("noi,...ni->...no", normalized_weight, input)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class ModSequential(nn.Sequential):
    # noinspection PyMethodOverriding
    def forward(self, input: torch.Tensor, context: torch.Tensor):
        for module in self:
            input = module(input, context)
        return input


class StatelessWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super(StatelessWrapper, self).__init__()
        self.module = module

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        return self.module(input)


class ModuleGroup(nn.Module):
    def __init__(
        self,
        module_specs: Mapping[str, Mapping[str, Union[List[str], nn.Module]]],
        flatten_output: bool = False,
    ):
        super(ModuleGroup, self).__init__()
        self.flatten_output = flatten_output
        self.all_modules = nn.ModuleDict(
            {key: module_spec["module"] for key, module_spec in module_specs.items()}
        )
        self.arguments = {
            key: (
                [module_spec["arguments"]]
                if isinstance(module_spec["arguments"], str)
                else list(module_specs["arguments"])
            )
            for key, module_spec in module_specs.items()
        }

    def new_module(
        self,
        key: str,
        module: nn.Module,
        arguments: List[str],
    ) -> "ModuleGroup":
        if isinstance(arguments, str):
            arguments = [arguments]
        else:
            arguments = list(arguments)
        self.all_modules.update({key: module})
        self.arguments.update({key: arguments})
        return self

    def forward(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = {}
        for key, module in self.all_modules.items():
            # Compile all the inputs
            module_inputs = [inputs[arg] for arg in self.arguments[key]]
            # Run the module
            output = module(*module_inputs)
            # Write to output
            outputs[key] = output
        if self.flatten_output:
            outputs = flatten_dict(outputs, sep="/")
        return outputs


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def drop_path(
    drop_prob: float,
    old: torch.Tensor,
    new: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    training: bool = True,
):
    """
    Implements stochastic depth, partially swiped from timm.
    Assumes that new = old + f(old), and applies new = old + drop(f(old)).
    """
    # Super fast codepath
    if not training and new is not None:
        return new

    if residual is None:
        assert new is not None
        residual = new - old
    else:
        assert new is None

    if not training:
        return old + residual

    keep_prob = 1 - drop_prob
    shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=residual.dtype, device=residual.device
    )
    random_tensor.floor_()  # binarize
    dropped_residual = residual.div(keep_prob) * random_tensor

    dropped_new = old + dropped_residual
    return dropped_new


def batched_map(fn: Callable, tensors: List[torch.Tensor]):
    """
    Like map, but batched. Concats the tensors along the 0-th axis,
    applies the fn, and then unconcats.
    """
    num_tensors = len(tensors)
    tensors = torch.cat(tensors, dim=0)
    tensors = fn(tensors)
    # Unconcat
    tensors = torch.chunk(tensors, num_tensors, dim=0)
    return list(tensors)


@lru_cache(maxsize=10)
def get_non_diag_mask(num_channels: int, device: Union[str, torch.device]):
    return ~torch.eye(num_channels, device=device, dtype=torch.bool)


def to_device(x, device, non_blocking=False):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    elif isinstance(x, (list, tuple)):
        # noinspection PyArgumentList
        return [to_device(_x, device, non_blocking=non_blocking) for _x in x]
    elif isinstance(x, dict):
        # noinspection PyArgumentList
        return {
            key: to_device(val, device, non_blocking=non_blocking)
            for key, val in x.items()
        }
    else:
        raise TypeError(f"Can't ship object of type {type(x)} to device.")


def as_2d(t: torch.Tensor, to_numpy: bool = False) -> Union[torch.Tensor, np.ndarray]:
    # Reshape to 2D
    if t.ndim > 2:
        t = rearrange(t, "... c -> (...) c")
    if to_numpy:
        t = t.data.cpu().numpy()
    return t


def reduce_tensor(
    x: torch.Tensor, mode: str = "mean", dim: Optional[Union[int, Iterable[int]]] = None
):
    if x.dim() > 0:
        if mode == "mean":
            x = x.mean(dim=dim)
        elif mode == "sum":
            x = x.sum(dim=dim)
        elif mode == "max":
            x = x.max(dim=dim)[0]
        elif mode == "none":
            pass
        elif mode == "select-0":
            # This selects the first element along each dim.
            sl = [slice(None)] * x.dim()
            dim = [dim] if isinstance(dim, int) else dim
            for d in dim:
                sl[d] = 0
            x = x[sl]
    return x


def gather(
    tensor: torch.Tensor,
    world_size: int,
    rank: Optional[int] = None,
    preserve_gradients: bool = False,
) -> List[torch.Tensor]:
    containers = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.barrier()
    dist.all_gather(containers, tensor)
    if preserve_gradients:
        # If rank is provided, use it; if not, infer it from distributed
        if rank is None:
            assert dist.is_initialized(), "Distributed is not initialized."
            rank = dist.get_rank()
        containers[rank] = tensor.clone()
    return containers


def gather_reduce(
    tensor: torch.Tensor,
    world_size: int,
    reduction: str = "mean",
    rank: Optional[int] = None,
    preserve_gradients: bool = False,
):
    containers = gather(
        tensor, world_size, rank=rank, preserve_gradients=preserve_gradients
    )
    pre_reduced = torch.stack(containers)
    if reduction == "mean":
        return pre_reduced.mean(0)
    elif reduction == "sum":
        return pre_reduced.sum(0)
    elif reduction == "stack":
        return pre_reduced
    elif reduction == "cat":
        if pre_reduced.dim() > 1:
            return rearrange(pre_reduced, "worker batch ... -> (worker batch) ...")
        else:
            return pre_reduced
    else:
        raise NotImplementedError(f"Reduction mode {reduction} not implemented.")


def sync_values(
    value,
    reduction: str,
    device: Union[str, torch.device],
    world_size: int,
    rank: Optional[int] = None,
    preserve_gradients: Optional[bool] = False,
):
    if not torch.is_tensor(value):
        if isinstance(value, np.ndarray):
            value_type = "ndarray"
        elif isinstance(value, (float, int)):
            value_type = "py_scalar"
        else:
            raise TypeError
        value = torch.tensor(value).to(device)
        original_device = "cpu"
    else:
        value_type = "tensor"
        original_device = value.device
        value = value.to(device)
    gathered_value = gather_reduce(
        value,
        world_size,
        reduction=reduction,
        rank=rank,
        preserve_gradients=preserve_gradients,
    )
    # Ship the tensor back to where it was
    gathered_value = gathered_value.to(original_device)
    if value_type != "tensor":
        if value_type == "ndarray":
            gathered_value = gathered_value.numpy()
        elif value_type == "py_scalar":
            gathered_value = gathered_value.item()
    return gathered_value


def straight_through_clamp(
    x: torch.Tensor,
    min_val: Union[torch.Tensor, float, None] = None,
    max_val: Union[torch.Tensor, float, None] = None,
):
    """Clamp, but with straight-through gradients."""
    with torch.no_grad():
        clamped = torch.clamp(x.data, min_val, max_val).sub_(x).detach_()
    return x + clamped


def check_keys(d, *expected_keys):
    for key in expected_keys:
        if key not in d:
            raise ValueError(f"Missing key {key} in {d}")
    return d


def define_aliases(d, **aliases):
    for key, value in aliases.items():
        d[key] = d[value]
    return d


def override(d, **backups):
    """
    If a key is in backups and d, then d[key] is kept.
    If it's only in backups but not in d, than backups[key] is kept.
    """
    return {**backups, **(d or {})}


def keep_valid(callable_, d):
    """Keep only the keys in d that are valid arguments of callable_."""
    if not callable(callable_):
        return d
    valid_parameters = [
        param.name
        for param in inspect.signature(callable_).parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    return {k: v for k, v in d.items() if k in valid_parameters}


def broadcast_update(
    d: Dict[str, Any],
    overrides: Dict[str, List],
    count: Optional[int] = None,
    sep: str = "/",
) -> List[Dict]:
    """
    This function updates copies of the dict d with the contents of overrides.

    For example, if d = {"a": 1, "b": 2}, and overrides = {"a": [7, 8, 9]}.
    The return is a list, [{"a": 7, "b": 2}, {"a": 8, "b": 2}, {"a": 9, "b": 2}].

    This function also supports nested dicts, but it's important that the keys
    do not contain the `sep` (default: "/").
    """
    # First step is to flatten overrides (which might have a nested structure)
    overrides = flatten_dict(overrides, sep=sep)
    if count is None:
        # Count the number of elements
        count = len(next(iter(overrides.values())))
    # Copy the dict to be broadcasted, `count` times
    broadcasted_copies = [deepcopy(d) for _ in range(count)]
    # Assemble the updates
    broadcasted_updates = [
        {key: values[idx] for key, values in overrides.items()} for idx in range(count)
    ]
    # Unflatten
    broadcasted_updates = [unflatten_dict(u, sep=sep) for u in broadcasted_updates]
    for idx in range(count):
        recursive_update(broadcasted_copies[idx], broadcasted_updates[idx])
    return broadcasted_copies


def keep_valid_backups(callable_, d, **backups):
    """
    Keep only the keys in backups that are valid arguments of callable_.
    In other words, if a key is contained in d but is not a valid argument
    of callable_, it will not be removed. But if a key in backups is not a
    valid argument, it will be removed.
    """
    # Get rid of things in backups that are not valid
    valid_backups = keep_valid(callable_, backups)
    return override(d, **valid_backups)


class DummyTorchDBTracer:
    @staticmethod
    def register_tracer_probes(*args):
        def decorator(cls):
            return cls

        return decorator

    @staticmethod
    def track(module, name, tensor):
        return tensor


def batch_dicts(*dicts, dim=0):
    return {key: torch.cat([d[key] for d in dicts], dim=dim) for key in dicts[0].keys()}


class ScheduledHyperParameter(nn.Module):
    def __init__(
        self, initial_value: Union[float, torch.Tensor], schedule_spec: str = None
    ):
        super(ScheduledHyperParameter, self).__init__()
        self.schedule_spec = schedule_spec
        self.register_buffer("value", torch.as_tensor(initial_value))
        self.register_buffer("progress", torch.tensor(0.0))

    def forward(self) -> torch.Tensor:
        self.update_value()
        return self.value

    @staticmethod
    def linear(
        progress: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        *,
        start: float,
        stop: float,
    ) -> torch.Tensor:
        return start * (1.0 - progress) + stop * progress

    def update_value(self):
        if self.schedule_spec is None:
            # Nothing to do
            return
        # Figure out the schedule
        schedule_name = self.schedule_spec[: self.schedule_spec.index("(")]
        schedule_kwargs = eval(self.schedule_spec.replace(schedule_name, "dict"))
        schedule_fn = getattr(self, schedule_name)
        # Evaluate the value and set it
        value = schedule_fn(self.progress, self.value, **schedule_kwargs)
        self.value.copy_(
            torch.as_tensor(value, device=self.value.device, dtype=self.value.dtype)
        )
        return self

    def update_progress(self, current: float, total: float):
        progress = current / total
        self.progress.copy_(
            torch.tensor(
                progress, device=self.progress.device, dtype=self.progress.dtype
            )
        )
        return self

    def update(self, current: Optional[float] = None, total: Optional[float] = None):
        if current is not None:
            assert total is not None
            self.update_progress(current, total)
        self.update_value()
        return self

    @classmethod
    def set_progress(cls, current: float, total: float) -> Callable:
        """
        The idea is to call `model.apply(ScheduledHyperParameter.set_progress(...))`.
        """

        def progress_setter(m):
            if isinstance(m, cls):
                m.update(current=current, total=total)

        return progress_setter
