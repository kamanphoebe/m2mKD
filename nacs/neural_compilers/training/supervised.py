import io
import math
import os.path
from argparse import Namespace
from contextlib import suppress
from copy import deepcopy
from typing import Optional, Union, Dict

import torch.cuda.amp
from torch.utils.data._utils.collate import default_collate

from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
from speedrun.logging.wandb import read_wandb_entity
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, ModelEma, accuracy
from torch import nn
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    DataLoader,
)
import wandb

import neural_compilers.model as nc_models
from neural_compilers.utils import to_device, ScheduledHyperParameter

#from neural_compilers.data import DATASET_PATH_REGISTRY, CACHE_PATH_REGISTRY
from neural_compilers.objectives.regularization import (
    SignatureHingeRepulsion,
    WeightedSumLoss,
    DistillationLoss,
    SignatureDistributionRegularization,
    StochasticBlockModelRegularization,
)
from neural_compilers.training.deit.datasets import build_dataset
from neural_compilers.training.deit.samplers import RASampler
from neural_compilers.training.deit.utils import MetricLogger, AccumulativeScaler
from neural_compilers.training.wormulon import WormulonMixin, get_slurm_job_id


class SupervisedTrainer(WormulonMixin, WandBMixin, IOMixin, BaseExperiment):
    WANDB_ENTITY = read_wandb_entity()
    WANDB_PROJECT = "m2mkd-nac"
    WANDB_SETTINGS = wandb.Settings(start_method="fork")
    # WANDB_SETTINGS = wandb.Settings(start_method="thread")

    def __init__(self, skip_setup: bool = False):
        super(SupervisedTrainer, self).__init__()
        if not skip_setup:
            if self.in_distributed_environment:
                self.distributed_setup()
            else:
                self.auto_setup()

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def set_step(self, step: int):
        self._step = step

    def _build(self):
        self._build_dataloaders()
        self._build_teacher()
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._build_criterion()
        self._setup_amp()
        self._resume_if_required()
        self._build_logging()

    def setup_dataset_kwargs(self):
        # Make the kwargs to call deit's build_dataset with
        dataset_kwargs = deepcopy(self.get("data/dataset/kwargs"))
        if "data_set" not in dataset_kwargs:
            dataset_kwargs["data_set"] = self.get("data/dataset/name")
        # Auto-read path if available
        """
        if (
            "data_path" not in dataset_kwargs
            and DATASET_PATH_REGISTRY.get(dataset_kwargs["data_set"]) is not None
        ):
            dataset_kwargs["data_path"] = DATASET_PATH_REGISTRY.get(
                dataset_kwargs["data_set"]
            )
        # Use a cached dataset if possible
        if (
            "cache_data_path" not in dataset_kwargs
            and CACHE_PATH_REGISTRY.get(dataset_kwargs["data_set"]) is not None
        ):
            dataset_kwargs["cache_data_path"] = CACHE_PATH_REGISTRY[
                dataset_kwargs["data_set"]
            ]
        """

        if "IMNET" in self.get("data/dataset/name"):
            # These keys need to be in data/dataset/kwargs; explicit is better
            # than implicit goddammit
            # fmt: off
            expected_keys = [
                "data_path", "input_size", "color_jitter", "aa",
                "train_interpolation", "reprob", "remode", "recount",
            ]
            # fmt: on
            for key in expected_keys:
                assert key in dataset_kwargs, (
                    f"Missing key {key} " f"in data/dataset/kwargs."
                )
        return dataset_kwargs

    def _build_dataloaders(self):
        # ----------------
        # Build the training / validation dataset
        dataset_kwargs = self.setup_dataset_kwargs()
        train_dataset, num_classes = build_dataset(
            is_train=True,
            args=Namespace(**dataset_kwargs),
            use_transform=dataset_kwargs.get("use_transform", True),
        )
        val_dataset, _ = build_dataset(
            is_train=False,
            args=Namespace(**dataset_kwargs),
            use_transform=dataset_kwargs.get("use_transform", True),
        )
        # This needs to be done in order for mixup to work
        self.set("data/dataset/num_classes", num_classes)
        # ----------------
        # Build the OOD dataset if required
        if self.get("data/ood_dataset", None) is not None:
            ood_dataset_kwargs = deepcopy(self.get("data/ood_dataset/kwargs", {}))
            if "data_set" not in ood_dataset_kwargs:
                ood_dataset_kwargs["data_set"] = self.get("data/ood_dataset/name")
            """
            if "data_path" not in ood_dataset_kwargs:
                data_path = DATASET_PATH_REGISTRY.get(ood_dataset_kwargs["data_set"])
                assert data_path is not None
                ood_dataset_kwargs["data_path"] = data_path
            """
            if "input_size" in ood_dataset_kwargs:
                assert ood_dataset_kwargs["input_size"] == dataset_kwargs["input_size"]
            else:
                ood_dataset_kwargs["input_size"] = dataset_kwargs["input_size"]
            ood_dataset, _ = build_dataset(
                is_train=False, args=Namespace(**ood_dataset_kwargs)
            )
        else:
            ood_dataset = None
        # ----------------
        # Build the sampler
        if self.get("data/sampler/repeated_augments") is not None:
            num_repeats = self.get("data/sampler/repeated_augments")
            # Though it might not appear that way, but RASampler should also
            # work in the not-distributed setting. In this case:
            #   self.rank = 0
            #   self.world_size = 1
            train_sampler = RASampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                num_repeats=num_repeats,
            )
        else:
            if self.is_distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                )
            else:
                train_sampler = RandomSampler(train_dataset)
        if self.is_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            if ood_dataset is not None:
                ood_sampler = DistributedSampler(
                    ood_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            else:
                ood_sampler = None
        else:
            val_sampler = SequentialSampler(val_dataset)
            if ood_dataset is not None:
                ood_sampler = SequentialSampler(ood_dataset)
            else:
                ood_sampler = None
        # ----------------
        # Build the loaders
        train_loader_kwargs = deepcopy(self.get("data/loader/kwargs"))
        train_loader_kwargs.update(dict(drop_last=True))
        assert "batch_size" in train_loader_kwargs
        if self.get("training/experiment_type", "pretrain") == "few-shot":
            train_loader_kwargs["batch_size"] = int((self.get("data/dataset/kwargs/k_way") 
                                                 * self.get("data/dataset/kwargs/n_shot")) / 4)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=getattr(train_dataset, "collate_fn", default_collate),
            **train_loader_kwargs,
        )
        val_loader_kwargs = dict(train_loader_kwargs)
        if self.get("training/experiment_type", "pretrain") == "few-shot":
            val_loader_kwargs["batch_size"] = int((self.get("data/dataset/kwargs/k_way") 
                                                 * self.get("data/dataset/kwargs/val_sample")) / 4)
        else:
            val_loader_kwargs["batch_size"] *= self.get(
                "data/loader/val_batch_size_multiplier", 1.0
            )
            val_loader_kwargs["batch_size"] = round(val_loader_kwargs["batch_size"])
        val_loader_kwargs["drop_last"] = False
        self.val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            collate_fn=getattr(val_dataset, "collate_fn", default_collate),
            **val_loader_kwargs,
        )
        if ood_dataset is not None:
            assert ood_sampler is not None
            self.ood_loader = DataLoader(
                ood_dataset,
                sampler=ood_sampler,
                **val_loader_kwargs,
            )
        else:
            self.ood_loader = None

    def _build_teacher(self):
        if not self.get("distillation/teacher/name", ""):
            self.teacher = None
            return self
        teacher = create_model(
            self.get("distillation/teacher/name"),
            pretrained=False,
            **self.get("distillation/teacher/kwargs", {})
        )
        ckpt = torch.load(self.get("distillation/teacher/checkpoint_path"), map_location='cpu')
        # for i, module in enumerate(teacher.stages):
        #     module.load_state_dict(ckpt['model_stages'][i])
        teacher.load_state_dict(ckpt['model'])
        teacher.to(self.device)
        teacher.eval()
        self.teacher = teacher
        return self

    def _build_model(self):
        model_cls = getattr(nc_models, self.get("model/name"))
        model = model_cls(**self.get("model/kwargs"))
        model.to(self.device)

        # Prep model EMA
        if self.get("training/ema/use", False):
            self.model_ema = ModelEma(
                model,
                decay=self.get("training/ema/decay", ensure_exists=True),
                device=self.get("training/ema/device", ""),
            )
        else:
            self.model_ema = None
        # self.model = self.wrap_model(
        #     model,
        #     find_unused_parameters=self.get(
        #         "model/find_unused_parameters_in_ddp", False
        #     ),
        #     set_static_graph=self.get("model/set_static_graph", False),
        # )
        # find_unused_parameters = True if self.resume_from_checkpoint is None else False
        find_unused_parameters = False
        if self.get("training/experiment_type", "pretrain") == "few-shot":
            checkpoint = torch.load(self.get("training/checkpoint_path"), map_location='cpu')
            # Load weights except for output tokernizer.
            filter_checkpoint = {}
            for key, val in checkpoint["model"].items():
                if "to_logits" in key:
                    continue
                filter_checkpoint[key] = val
            model.load_state_dict(filter_checkpoint, strict=False)
            for name, module in model.named_modules():
                if "to_logits" in name:
                    module.requires_grad_(True)
                else:
                    module.requires_grad_(False)
        self.model = self.wrap_model(
            model,
            find_unused_parameters=find_unused_parameters,
            set_static_graph=self.get("model/set_static_graph", False),
        )
        return self

    def _build_optimizer(self):
        # Build optimizer
        optim_kwargs = deepcopy(self.get("optimizer/kwargs"))
        optim_kwargs["optimizer_name"] = self.get("optimizer/name")
        # Scale the learning rate by the batch and world size
        if self.get("training/experiment_type", "pretrain") == "few-shot":
            scaled_lr = optim_kwargs["lr"]
        else:
            scaled_lr = (
                optim_kwargs["lr"]
                * self.get("data/loader/kwargs/batch_size", ensure_exists=True)
                * self.world_size
                / 512.0
            )
        # We'll need to get rid of the lr kwarg, because it's set inside
        # timm's create_optimizer ...
        del optim_kwargs["lr"]
        # ... instead, we'll need learning_rate kwarg.
        optim_kwargs["learning_rate"] = scaled_lr
        # We need to make sure that the weight decay is being applied
        # to the right params.
        if (
            "weight_decay" in optim_kwargs
            and optim_kwargs["weight_decay"] > 0.0
            and self.get("optimizer/no_weight_decay_filter") is not None
        ):
            weight_decay = optim_kwargs["weight_decay"]
            # Make the param groups, one with and the other without weight decay
            no_weight_decay_filter_fn = eval(
                f"lambda name, param: "
                f"{self.get('optimizer/no_weight_decay_filter')}"
            )
            no_weight_decay_group = [
                param
                for name, param in self.unwrap_model(self.model).named_parameters()
                if no_weight_decay_filter_fn(name, param)
            ]
            no_weight_decay_group_ids = [id(p) for p in no_weight_decay_group]
            weight_decay_group = [
                param
                for param in self.unwrap_model(self.model).parameters()
                if id(param) not in no_weight_decay_group_ids
            ]
            param_list = [
                {"params": no_weight_decay_group, "weight_decay": 0.0},
                {"params": weight_decay_group, "weight_decay": weight_decay},
            ]
            # Set the overall weight_decay term to 0 (since its now
            # being handled by the param groups)
            optim_kwargs["weight_decay"] = 0.0

            # This is awkward, but timm wants a model it can call .parameters() on.
            class DummyModel:
                def parameters(self):
                    return param_list

            model = DummyModel()
        else:
            model = self.unwrap_model(self.model)
        self.optimizer = create_optimizer_v2(model, **optim_kwargs)
        return self

    def _build_scheduler(self):
        # Build scheduler
        scheduler_kwargs = deepcopy(self.get("scheduler/kwargs"))
        if scheduler_kwargs is None:
            self.scheduler = None
            return self
        if "sched" not in scheduler_kwargs:
            scheduler_kwargs["sched"] = self.get("scheduler/name")
        if "epochs" not in scheduler_kwargs:
            scheduler_kwargs["epochs"] = self.num_epochs
        # TODO: Confirm if min-lr should be adjusted. See this issue:
        #  https://github.com/facebookresearch/deit/issues/144
        scaled_min_lr = (
            scheduler_kwargs["min_lr"]
            * self.get("data/loader/kwargs/batch_size", ensure_exists=True)
            * self.world_size
            / 512
        )
        scheduler_kwargs["min_lr"] = scaled_min_lr
        scheduler, _ = create_scheduler(Namespace(**scheduler_kwargs), self.optimizer)
        self.scheduler = scheduler
        return self

    def _build_criterion(self):
        if self.get("criterion/mixup_and_cutmix/use", True):
            mixup_and_cutmix_kwargs = deepcopy(
                self.get("criterion/mixup_and_cutmix/kwargs", {})
            )
            kwarg_updates = dict(
                label_smoothing=self.get("criterion/label_smoothing", 0.1),
                num_classes=self.num_classes,
            )
            mixup_and_cutmix_kwargs.update(kwarg_updates)
            self.mixup_fn = Mixup(**mixup_and_cutmix_kwargs)
        else:
            self.mixup_fn = None
        # Classification objective
        if self.mixup_fn is not None:
            criterion = SoftTargetCrossEntropy()
        else:
            if self.get("criterion/label_smoothing", 0.1) != 0.0:
                criterion = LabelSmoothingCrossEntropy()
            else:
                criterion = nn.CrossEntropyLoss()
        criterion = WeightedSumLoss(simplified_interface=False).add_loss(
            criterion, 1.0, "classification"
        )
        # Sparsity objective
        if self.get("criterion/graph_sparsity/use", False):
            graph_sparsity_type = self.get(
                "criterion/graph_sparsity/type", "signature_hinge_repulsion"
            )
            if graph_sparsity_type == "signature_hinge_repulsion":
                criterion.add_loss(
                    SignatureHingeRepulsion(
                        model=self.unwrap_model(self.model),
                        hinge=self.get(
                            "criterion/graph_sparsity/hinge", ensure_exists=True
                        ),
                        **self.get("criterion/graph_sparsity/kwargs", {}),
                    ),
                    self.get("criterion/graph_sparsity/weight", 1.0),
                    "graph_sparsity",
                )
            elif graph_sparsity_type == "signature_distribution_regularization":
                criterion.add_loss(
                    SignatureDistributionRegularization(
                        model=self.unwrap_model(self.model),
                        **self.get("criterion/graph_sparsity/kwargs", {}),
                    ).to(self.device),
                    self.get("criterion/graph_sparsity/weight", 1.0),
                    "graph_sparsity",
                )
            elif graph_sparsity_type == "stochastic_block_model_regularization":
                criterion.add_loss(
                    StochasticBlockModelRegularization(
                        model=self.unwrap_model(self.model),
                        **self.get("criterion/graph_sparsity/kwargs", {}),
                    ).to(self.device),
                    self.get("criterion/graph_sparsity/weight", 1.0),
                    "graph_sparsity",
                )
            else:
                raise ValueError
        self.criterion = DistillationLoss(base_criterion=criterion, 
                                          teacher_model=self.teacher, 
                                          distillation_type=self.get("distillation/type", 'none'),
                                          **self.get("distillation/kwargs", {}))
        # Set the criterion
        # self.criterion = criterion

    def _setup_amp(self):
        if self.get("training/use_amp", True):
            self.amp_autocast = torch.cuda.amp.autocast
            if self.get("training/accumulation_steps", 1) > 1:
                self.loss_scaler = AccumulativeScaler(
                    self.get("training/accumulation_steps")
                )
            else:
                self.loss_scaler = NativeScaler()
        else:
            self.amp_autocast = suppress
            self.loss_scaler = None
            raise ValueError(
                "As fate would have it, " "not using AMP is not supported."
            )
        return self

    def _resume_if_required(self):
        if self.resume_from_checkpoint is None:
            return self
        if self.resume_from_checkpoint == "latest":
            # Load the latest ckpt and go
            checkpoint = self.load(load_latest=True)
        elif self.resume_from_checkpoint == "best":
            checkpoint = self.load(load_best=True)
        elif (
            isinstance(self.resume_from_checkpoint, bool)
            and self.resume_from_checkpoint
        ):
            # This supports an older way of resuming training
            checkpoint = self.load(load_latest=True)
        else:
            checkpoint = self.load(checkpoint_path=self.resume_from_checkpoint)
        # Set the starting epoch num
        self.set("training/start_epoch", checkpoint["epoch"] + 1)
        # Done
        return self

    def _build_logging(self):
        # We also want to log the slurm job id to wandb
        self.set("speedrun_meta/slurm_job_id", get_slurm_job_id())
        self.set("speedrun_meta/experiment_name", self.experiment_name)
        self.set("speedrun_meta/base_directory", self.experiment_base_directory)
        self.num_iterations_with_nan = 0
        if self.is_chief:
            self.initialize_wandb(resume=(self.resume_from_checkpoint is not None))
            if self.get("wandb/watch", False):
                self.wandb_watch(self.unwrap_model(self.model))
        return self

    @property
    def ood_data_available(self):
        return self.ood_loader is not None

    @property
    def num_epochs(self) -> int:
        return self.get("training/num_epochs", 300)

    @property
    def start_epoch(self) -> int:
        return self.get("training/start_epoch", 0)

    @property
    def num_classes(self) -> int:
        return self.get("data/dataset/num_classes", ensure_exists=True)

    @property
    def resume_from_checkpoint(self) -> Union[str, None]:
        return self.get_arg("resume", self.get("training/resume", None))

    def compute_accuracies(self, output, target, loss, metrics):
        if "NLVR" in self.get("data/dataset/name"):
            acc1 = (output.argmax(dim=1) == target).sum() / target.numel()
            metrics.update(loss=loss.item(), acc1=acc1.item())
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            metrics.update(loss=loss.item())
            #metrics.meters["acc1"].update(acc1.item(), n=images.shape[0])
            #metrics.meters["acc5"].update(acc5.item(), n=images.shape[0])
            metrics.meters["acc1"].update(acc1.item(), n=target.shape[0])
            metrics.meters["acc5"].update(acc5.item(), n=target.shape[0])

        acc1 = (output.argmax(dim=1) == target).sum() / target.numel()
        metrics.update(loss=loss.item(), acc1=acc1.item())
        return metrics

    def before_training_loop(self):
        # Set the schedule straight
        if self.scheduler is not None and self.start_epoch > 0:
            self.scheduler.step(self.start_epoch)
        return self

    def prepare_for_epoch(self, epoch: int):
        # Set epoch (this is not redundant if training is resumed)
        self.set_epoch(epoch)
        # Set the data loader seed. If this is not done, all epochs see
        # the same order of samples.
        if (
            self.get("data/sampler/repeated_augments") is not None
            or self.is_distributed
        ):
            self.train_loader.sampler.set_epoch(epoch)
        # Tell the schedulers in model and criterion what epoch it is
        self.model.apply(
            ScheduledHyperParameter.set_progress(current=epoch, total=self.num_epochs)
        )
        self.criterion.apply(
            ScheduledHyperParameter.set_progress(current=epoch, total=self.num_epochs)
        )
        # Log things around the model
        self.log_model()

    def handle_nan(self, loss, samples, targets):
        # Die if loss is infinite
        if not math.isfinite(loss.item()):
            self.num_iterations_with_nan += 1
            if self.num_iterations_with_nan > self.get_arg("nan_tolerance", 0):
                extra_things_to_save = dict(
                    samples=samples.cpu(), targets=targets.cpu()
                )
                # Save the borked checkpoint for post-mortem
                self.save(
                    for_post_mortem=True, extra_things_to_save=extra_things_to_save
                )
                # We piggy back on the requeue logic to have the run restarted
                # by the sentry.
                self.request_requeue(reason="nan_in_loss")
                raise ValueError(
                    f"NaN (#{self.num_iterations_with_nan}) encountered during training"
                    f" at epoch = {self.epoch} and step = {self.step}."
                )

    def train_epoch(self):
        metrics = MetricLogger()
        self.model.train()
        self.num_iterations_with_nan = 0

        for iter_num, (samples, targets) in enumerate(self.train_loader):
            if self.get("training/ship_to_device", True):
                samples = to_device(samples, device=self.device, non_blocking=True)
            targets = to_device(targets, device=self.device, non_blocking=True)
            if self.mixup_fn is not None:
                samples, targets = self.mixup_fn(samples, targets)
            # amp_autocast should be torch.cuda.amp.autocast
            with self.amp_autocast():
                outputs = self.model(samples)
                loss_dict = self.criterion(samples, outputs, targets)
            loss = loss_dict["loss"]

            self.handle_nan(loss, samples, targets)
            self.optimizer.zero_grad()

            # The loss_scaler does the optimizer step (together with
            # gradient clipping, if required)
            self.loss_scaler(
                loss,
                self.optimizer,
                clip_grad=self.get("training/clip_grad"),
                parameters=self.model.parameters(),
                create_graph=getattr(self.optimizer, "is_second_order", False),
            )

            torch.cuda.synchronize()
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            metrics.update(
                **{name: loss_val.item() for name, loss_val in loss_dict.items()}
            )
            if "NLVR" in self.get("data/dataset/name"):
                metrics = self.compute_accuracies(outputs, targets, loss, metrics)

            self.log_while_training(metrics, lr=self.optimizer.param_groups[0]["lr"])
            self.next_step()

        metrics.synchronize_between_processes()
        return {k: meter.global_avg for k, meter in metrics.meters.items()}

    @torch.no_grad()
    def validate_epoch(self, ood: bool = False):
        # Select the right loader
        if ood:
            assert self.ood_data_available
            loader = self.ood_loader
        else:
            loader = self.val_loader
        if hasattr(loader.dataset, "prediction_transform"):
            prediction_transform = getattr(loader.dataset, "prediction_transform")
            assert callable(prediction_transform)
        else:
            prediction_transform = lambda x: x
        # Criterion and metrics
        criterion = nn.CrossEntropyLoss()
        metrics = MetricLogger()
        # Set to eval and go
        self.model.eval()
        for samples, target in loader:
            if self.get("training/ship_to_device", True):
                samples = to_device(samples, device=self.device, non_blocking=True)
            target = to_device(target, device=self.device, non_blocking=True)
            # Evaluate with AMP
            with self.amp_autocast():
                output = self.model(samples)
                output = prediction_transform(output)
                loss = criterion(output, target)
            # Evaluate accuracies
            metrics = self.compute_accuracies(output, target, loss, metrics)
        # Sync between processes
        metrics.synchronize_between_processes()
        self.log_after_validation(metrics, prefix=("validation" if not ood else "ood"))
        return {k: meter.global_avg for k, meter in metrics.meters.items()}

    def after_training_epoch(self, epoch: int, train_stats: Optional[dict] = None):
        # Save
        self.save(is_latest=True)
        if self.scheduler is not None:
            self.scheduler.step(epoch + 1)
        return self

    def after_validating_epoch(self, epoch: int, val_stats: Optional[dict] = None):
        # if int(os.environ["LOCAL_RANK"]) != 0:
        #     return
        if val_stats is not None:
            self.write_to_cache("current_val_stats", val_stats)
            self.save_if_best(val_stats)
        self.save_periodically(epoch)
        return self

    def after_ood_evaluation(self, epoch: int, ood_stats: Optional[dict] = None):
        if ood_stats is not None:
            self.write_to_cache("current_ood_stats", ood_stats)
        return self

    def finish_epoch(self, epoch: int):
        # Increment epoch counter
        self.set_epoch(epoch + 1)

    @register_default_dispatch
    def train(self):
        self._build()
        self.before_training_loop()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.prepare_for_epoch(epoch)
            # Train for an epoch
            train_stats = self.train_epoch()
            self.after_training_epoch(epoch, train_stats)
            # OOD first, because after_validating_epoch we make a checkpoint
            if self.ood_data_available:
                ood_stats = self.validate_epoch(ood=True)
                self.after_ood_evaluation(epoch, ood_stats)
            # Evaluate
            val_stats = self.validate_epoch()
            self.after_validating_epoch(epoch, val_stats)
            self.finish_epoch(epoch)

    def save_if_best(self, current_val_stats: dict):
        is_best = current_val_stats["acc1"] > self.read_from_cache("best_acc1", 0.0)
        if not is_best:
            return
        # If we're here, we best
        self.write_to_cache("best_acc1", current_val_stats["acc1"])
        # Save
        self.save(is_best=True)
        return self

    def save_periodically(self, epoch):
        if self.get("training/checkpoint_every", None) is None:
            return
        if (epoch % self.get("training/checkpoint_every")) == 0:
            self.save(
                checkpoint_path=os.path.join(
                    self.checkpoint_directory, f"checkpoint_epoch_{epoch}.pt"
                )
            )
        return self

    def save(
        self,
        checkpoint_path: Optional[str] = None,
        is_latest: bool = False,
        is_best: bool = False,
        for_post_mortem: bool = False,
        extra_things_to_save: Optional[dict] = None,
    ):
        if self.is_chief or for_post_mortem:
            if for_post_mortem:
                # We have things to do here, but first some asserts
                assert not is_latest and not is_best
        else:
            # Nothing to do in this function.
            return
        if checkpoint_path is None:
            if is_latest:
                checkpoint_path = os.path.join(
                    self.checkpoint_directory, "checkpoint_latest.pt"
                )
            elif is_best:
                checkpoint_path = os.path.join(
                    self.checkpoint_directory, "checkpoint_best.pt"
                )
            elif for_post_mortem:
                checkpoint_path = os.path.join(
                    self.checkpoint_directory,
                    f"checkpoint_post_mortem_rank_{self.rank}.pt",
                )
            else:
                checkpoint_path = self.checkpoint_path
        # FIXME: The state_dict of criterion needs saving (esp. when using distreg), but that
        #  state_dict also contains things that are in model.state_dict().
        checkpoint = dict(
            model=self.unwrap_model(self.model).state_dict(),
            optimizer=self.optimizer.state_dict(),
            lr_scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
            model_ema=(
                self.model_ema.ema.state_dict() if self.model_ema is not None else None
            ),
            scaler=(
                self.loss_scaler.state_dict() if self.loss_scaler is not None else None
            ),
            best_acc1=self.read_from_cache("best_acc1", 0.0),
            epoch=self.epoch,
            step=self.step,
            current_val_stats=self.read_from_cache("current_val_stats", {}),
            current_ood_stats=self.read_from_cache("current_ood_stats", {}),
            **(extra_things_to_save or {}),
        )
        torch.save(checkpoint, checkpoint_path)
        return self

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        load_latest: bool = False,
        load_best: bool = False,
    ):
        if checkpoint_path is None:
            if load_latest:
                file_name = "checkpoint_latest.pt"
            elif load_best:
                file_name = "checkpoint_best.pt"
            else:
                raise ValueError(
                    "Checkpoint path could not be auto-inferred, please provide one."
                )
            checkpoint_path = os.path.join(self.checkpoint_directory, file_name)
        assert checkpoint_path is not None and os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Load in the model (don't worry about the device,
        # that gets handled inside the load_state_dict method)
        self.unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if self.model_ema is not None:
            # timm's model_ema wants to load from a file, but we aint
            # got no file with just the ema checkpoint
            file_in_mem = io.BytesIO()
            torch.save(checkpoint["model_ema"], file_in_mem)
            file_in_mem.seek(0)
            # noinspection PyProtectedMember
            self.model_ema._load_checkpoint(file_in_mem)
        if self.loss_scaler is not None:
            self.loss_scaler.load_state_dict(checkpoint["scaler"])
        self.write_to_cache("best_acc1", checkpoint["best_acc1"])
        self.set_epoch(checkpoint["epoch"] + 1)
        self.set_step(checkpoint["step"] + 1)
        # Done
        return checkpoint

    def log_while_training(self, metrics: MetricLogger, **extra_metrics):
        if not self.is_chief:
            return self
        payload = {}
        for key in metrics.meters:
            payload[f"training/{key}"] = metrics.meters[key].avg
        payload.update({f"training/{k}": v for k, v in extra_metrics.items()})
        if self.log_wandb_now:
            self.wandb_log(**payload)
        return self

    def log_after_validation(
        self, metrics: MetricLogger, prefix="validation", **extra_metrics
    ):
        if not self.is_chief:
            return self
        payload = {}
        for key in metrics.meters:
            payload[f"{prefix}/{key}"] = metrics.meters[key].global_avg
        payload.update({f"{prefix}/{k}": v for k, v in extra_metrics.items()})
        self.wandb_log(**payload)
        return self

    def log_model(self):
        # TODO: This should log the sparsity amount
        pass
