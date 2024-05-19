from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_lsj import dataloader


model = model_zoo.get_config("common/models/mask_rcnn_vmoedet.py").model

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.seed = 54321
train.output_dir = 'output'
train.checkpointer.max_to_keep = 2
train.checkpointer.period = 10000
train.eval_period = 20000
train.log_period = 200

dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.total_batch_size = 8
train.init_checkpoint = (
    "m2mKD/log_dir/FT_vmoe/finished_checkpoint.pth"
)


# Schedule
train.max_iter = 184375 * 8

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889*8, 177546*8],
        num_updates=train.max_iter,
    ),
    warmup_length=250*8 / train.max_iter,
    warmup_factor=0.001,
)


# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
