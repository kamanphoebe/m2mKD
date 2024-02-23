import datetime
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Union

import numpy as np
import torch.cuda
import torch.distributed as td
from speedrun.utils.yaml_utils import dump_yaml

from neural_compilers.utils import sync_values, gather


class SlurmSpec(object):
    @property
    def job_id(self):
        return os.getenv("SLURM_JOB_ID")

    @property
    def num_tasks(self):
        return int(os.getenv("SLURM_NTASKS", 1))

    @property
    def tasks_per_node(self):
        return self.num_tasks // self.num_nodes

    @property
    def local_id(self):
        return int(os.getenv("SLURM_LOCALID", 0))

    @property
    def num_nodes(self):
        return int(os.getenv("SLURM_NNODES", 1))

    @property
    def node_id(self):
        return int(os.getenv("SLURM_NODEID", 0))

    @property
    def rank(self):
        return (self.node_id * self.num_nodes) + self.local_id

    @property
    def world_size(self):
        return self.num_tasks

    @property
    def launch_node_ip_address(self):
        return os.getenv("SLURM_LAUNCH_NODE_IPADDR")

    @property
    def in_distributed_environment(self):
        return self.num_tasks > 1

    @property
    def device_id(self):
        return self.local_id

    @property
    def device(self):
        if torch.cuda.is_available():
            return f"cuda:{self.device_id}"
        else:
            return "cpu"

    def gather(self, tensor: torch.Tensor, preserve_gradients: bool = False):
        if self.in_distributed_environment:
            return gather(
                tensor,
                world_size=self.world_size,
                rank=self.rank,
                preserve_gradients=preserve_gradients,
            )
        else:
            # We simulate the output of gather.
            if preserve_gradients:
                return [tensor.clone()]
            else:
                return [torch.empty_like(tensor).copy_(tensor.data)]

    def sync_values(
        self,
        value,
        reduction: str,
        preserve_gradients: bool = False,
        raise_when_not_in_distributed_environment: bool = False,
    ):
        if self.in_distributed_environment:
            return sync_values(
                value=value,
                reduction=reduction,
                device=self.device,
                world_size=self.world_size,
                rank=self.rank,
                preserve_gradients=preserve_gradients,
            )
        else:
            if raise_when_not_in_distributed_environment:
                raise RuntimeError("Can't sync when not in distributed env.")
            else:
                return value


SLURM = SlurmSpec()


def get_slurm_job_id():
    return SLURM.job_id


class WormulonMixin(object):
    @property
    def in_distributed_environment(self):
        return int(os.environ["WORLD_SIZE"]) > 1
        # return SLURM.in_distributed_environment

    @property
    def job_uuid(self):
        # uuid = self.get_arg("uuid", os.getenv("SPEEDRUN_UUID"))
        uuid = "d5a4c8de-5cab-4db1-9e5b-5fc3cc23de59"
        assert uuid is not None
        return uuid

    @property
    def read_config_signal_path(self):
        return os.path.join(self.log_directory, f"read_config_{self.job_uuid}.signal")

    def mark_config_as_ready(self):
        # WARNING: exist_ok should ideally be False, because it's very much not
        # okay if the signal file already exists. But setting it to False leads
        # to mysterious errors when jobs are resumed after preemption (even though
        # they shouldn't), and we don't quite know why this file exist. Until we
        # figure that out, we'll play cowboy and set it to True.
        Path(self.read_config_signal_path).touch(exist_ok=True)
        return self

    def mark_config_as_unready(self):
        Path(self.read_config_signal_path).unlink(missing_ok=True)
        return self

    @property
    def config_is_ready(self):
        return Path(self.read_config_signal_path).exists()

    def wait_for_config_file(
        self, file_name: str = "train_config.yml", read: bool = True
    ):
        # Sleep for a while before querying the config file, just for good measure.
        time.sleep(5)
        while not self.config_is_ready:
            time.sleep(5)
        # Use the extra 5 seconds for the config to have been fully written out,
        # just in case
        time.sleep(5)
        if read:
            # Read it in now
            self.read_config_file(file_name=file_name)

    def dist_console(self, message: str):
        if self.get_arg("verbose", True):
            print(message, file=sys.stderr)
        return self

    def setup_chief(self):
        self.record_args()
        self.auto_setup(
            dump_configuration=self.get_arg("speedrun.dump_configuration", True)
        )
        self.mark_config_as_ready()
        return self

    def setup_padawan(self):
        self.record_args()
        self.parse_experiment_directory()
        # The chief writes out the config file, which the padawan reads
        self.wait_for_config_file(read=True)
        return self

    def distributed_setup(self):
        assert self.in_distributed_environment
        # WARNING: Don't read self.is_chief, it doesn't work because
        # is_distributed is not set yet.
        # if SLURM.rank == 0:
        if int(os.environ["RANK"]) == 0:
            self.setup_chief()
        else:
            self.setup_padawan()
        # WARNING: It's imperative that the line below remain where it is.
        self.set("distributed/is_distributed", True)
        self.init_distributed()
        self.register_signal_handlers()

    def init_distributed(self):
        # Set the environment variables for distributed
        # os.environ["MASTER_ADDR"] = SLURM.launch_node_ip_address
        # os.environ["MASTER_PORT"] = str(self.get_arg("distributed_port", 32914))
        try:
            td.init_process_group(
                # backend="gloo",
                backend="nccl",
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=int(os.environ["RANK"]),
            )
            self.dist_console("Successfully initialized distributed.")
            self.dist_console(f"Rank / World Size = {int(os.environ['RANK'])} / {int(os.environ['WORLD_SIZE'])}")
        except RuntimeError:
            # Print debug statements
            self.dist_console("RuntimeError when attempting to init process group.")
            self.dist_console(f"  MASTER_ADDR      = {os.environ['MASTER_ADDR']}")
            self.dist_console(f"  MASTER_PORT      = {os.environ['MASTER_ADDR']}")
            self.dist_console(f"  self.rank        = {self.rank}")
            self.dist_console(f"  SLURM.rank       = {SLURM.rank}")
            self.dist_console(f"  self.world_size  = {self.world_size}")
            self.dist_console(f"  SLURM.world_size = {SLURM.world_size}")
            self.dist_console("Traceback follows.")
            raise
        assert torch.cuda.is_available(), "Something is messed up."
        # torch.cuda.set_device(self.device_id)
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    @property
    def is_distributed(self):
        return self.get("distributed/is_distributed", False)

    @property
    def world_size(self):
        if self.is_distributed:
            # return SLURM.world_size
            return int(os.environ["WORLD_SIZE"])
        else:
            return 1

    @property
    def rank(self):
        if self.is_distributed:
            return int(os.environ["RANK"])
            # return SLURM.rank
        else:
            return 0

    @property
    def is_chief(self):
        if self.is_distributed:
            return self.rank == 0
        else:
            return True

    @property
    def device_id(self):
        if self.is_distributed:
            # return SLURM.device_id
            return int(os.environ["LOCAL_RANK"])
        else:
            return 1

    @property
    def device(self):
        if self.is_distributed:
            # return SLURM.device
            return f"cuda:{int(os.environ['LOCAL_RANK'])}"
        else:
            return self.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    def sync(self):
        if not self.is_distributed or not td.is_available():
            return
        td.barrier()

    def sync_values(
        self,
        value: Union[torch.Tensor, np.ndarray, int, float],
        reduction: str = "mean",
    ):
        if not self.is_distributed:
            return value
        return sync_values(
            value=value,
            reduction=reduction,
            device=self.device,
            world_size=self.world_size,
        )

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module
        else:
            return model

    def wrap_model(
        self,
        model: torch.nn.Module,
        find_unused_parameters: bool = False,
        set_static_graph: bool = False,
    ) -> Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]:
        if not self.is_distributed:
            return model
        if len(list(model.parameters())) == 0:
            # We don't wrap if there are no parameters
            return model
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.device_id],
            find_unused_parameters=find_unused_parameters,
        )
        if set_static_graph:
            # noinspection PyProtectedMember
            model._set_static_graph()
        return model

    def get_local_batch_size(self, global_batch_size: int) -> int:
        if not self.is_distributed:
            return global_batch_size
        assert (global_batch_size % self.world_size) == 0
        local_batch_size = global_batch_size // self.world_size
        return local_batch_size

    @property
    def requeue_request_directory(self):
        request_path = os.path.expanduser("~/.salvo_requeue_requests")
        os.makedirs(request_path, exist_ok=True)
        return request_path

    def request_requeue(self, reason: str = None):
        # If this happens, we'll need to write out enough info for
        # the job to be resumed by some daemon.
        if self.is_chief:
            request = {
                "experiment_directory": self.experiment_directory,
                "requeue_request_at": time.time(),
                "request_from_slurm_job_id": SLURM.job_id,
                "reason": reason,
            }
            requeue_request_file_name = (
                f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                f"_{uuid.uuid4().hex}.yml"
            )
            request_file_path = os.path.join(
                self.requeue_request_directory, requeue_request_file_name
            )
            dump_yaml(request, request_file_path)
            return request_file_path

    def preemption_panic(self):
        # If this happens, we'll need to write out enough info for
        # the job to be resumed by some daemon.
        if self.is_chief:
            self.request_requeue(reason="preemption")

    def register_signal_handlers(self):
        if self.is_chief:
            signal.signal(signal.SIGUSR1, self.preemption_panic)
