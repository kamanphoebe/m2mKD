# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import random
import os
import yaml
from glob import glob

from pathlib import Path

from deep_incubation.timm.data import Mixup
from deep_incubation.timm.models import create_model
from deep_incubation.timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from deep_incubation.timm.scheduler import create_scheduler
from deep_incubation.timm.optim import create_optimizer
from utils import NativeScaler

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, m2mKDLoss
from samplers import RASampler
import models
import utils

import vmoe
import nacstudent


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--model_ckpt_path', type=str)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher_divided_depths', default=None, type=int, nargs='+')
    parser.add_argument('--teacher_ckpt_path', type=str, default='')
    parser.add_argument('--meta', type=str, default='stitch_vit_meta')
    parser.add_argument('--meta_divided_depths', type=int, default=None, nargs='+')
    parser.add_argument('--meta_ckpt_path', default='', type=str)
    parser.add_argument('--distillation_type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--alpha_teacher', default=0.5, type=float)
    parser.add_argument('--distillation_tau', default=1.0, type=float, help="")
    # NACs student
    parser.add_argument('--student_cfg_path', default='./configs/nac_student_tinyimnet_config.yml', type=str)
    # VMoE student
    parser.add_argument('--moe_pos', type=int, default=None, nargs='+')
    parser.add_argument('--pre_stitch_pos', type=int, default=None, nargs='+')
    parser.add_argument('--post_stitch_pos', type=int, default=None, nargs='+')
    parser.add_argument('--num_expert', type=int, default=8)
    parser.add_argument('--stitch_dim', type=int, default=None)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR100-FS', 'CUB-FS', 'CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19', 'IMNET-R'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    
    parser.add_argument('--k_way', type=int)
    parser.add_argument('--n_shot', type=int)
    parser.add_argument('--val_sample', type=int)

    parser.add_argument('--output_dir', default='.',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_modules', default=False, type=bool)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--phase', type=str, choices=['PT', 'MT', 'FT', 'm2mKD', 'FS', 'E2E'], default='E2E')
    parser.add_argument('--divided_depths', type=int, default=None, nargs='+')  # depth for each module
    parser.add_argument('--idx', type=int, default=None)  # target module's index in MT
    parser.add_argument('--incubation_models', type=str, default='', nargs='+')  # incubated modules path, for AS

    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--use_amp', default=0, type=int)
    parser.add_argument('--save_ckpt_num', default=2, type=int)
    parser.add_argument('--save_freq', type=int, default=1)
    return parser


def main(args):

    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = None
    sampler_val = None
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    if args.phase == 'FS':
        args.batch_size = args.k_way * args.n_shot

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        shuffle=False if sampler_train else True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.model == 'nac':
        with open(args.student_cfg_path, 'r') as f:
            student_cfg = yaml.load(f, Loader=yaml.FullLoader)
        student_type = get_nac_student_type(args.idx)
        print(f"Creating model: {student_type}")
        model = create_model(student_type, stitch_dim=args.stitch_dim, **student_cfg['share_kwargs'], **student_cfg[student_type])
    elif 'vmoe' in args.model:
        print(f"Creating model: {args.model}")
        model = create_model(
            args.model,
            idx=args.idx,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            divided_depths=args.divided_depths,
            moe_pos=args.moe_pos,
            num_expert=args.num_expert,
            pre_stitch_pos=args.pre_stitch_pos,
            post_stitch_pos=args.post_stitch_pos,
            stitch_dim=args.stitch_dim,
        )
    else:
        print(f"Creating model: {args.model}")
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            divided_depths=args.divided_depths
        )
    

    if args.finetune:
        print(f'Loading checkpoint from {args.finetune}')
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        for i, (module, ckpt) in enumerate(zip(model.stages, checkpoint['model_stages'])):
            if i == 0:
                # interpolate position embedding
                pos_embed_checkpoint = ckpt['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = module.patch_embed.num_patches
                num_extra_tokens = module.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic',
                                                            align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                ckpt['pos_embed'] = new_pos_embed
            if args.phase == 'FS' and i == len(args.divided_depths) - 1:
                # Not loading classifier in full fine-tuning or few shot phase.
                ckpt.pop('head.weight')
                ckpt.pop('head.bias')
                module.load_state_dict(ckpt, strict=False)
            else:
                module.load_state_dict(ckpt)
    
    if args.phase == 'FS':
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True
                                                          if args.phase == 'MT' and args.idx != 0
                                                          else False,)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if utils.is_main_process():
        print('number of params:', n_parameters)

    if args.phase == 'MT':
        assert args.idx in range(len(model_without_ddp.stages))
        load_meta(args, model_without_ddp, model_is_meta=False)

        model2optimize = model_without_ddp.stages[args.idx]  # only optimize the target module in the hybrid model
        for i, module in enumerate(model_without_ddp.stages):
            if i < args.idx:
                for param in module.parameters():
                    param.requires_grad = False  # set the meta modules before the target module to no_grad
    else:
        model2optimize = model_without_ddp
        if args.phase == 'FT':
            model_assemb(args, model_without_ddp)
    
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    if args.phase == 'FS':
        linear_scaled_lr = args.lr
    else:
        linear_scaled_lr = args.lr * total_batch_size / 512.0
    args.lr = linear_scaled_lr
    print(
        f'Update freq: {args.update_freq}, scaled_lr: {linear_scaled_lr}, scale_rate: {args.batch_size * args.update_freq * utils.get_world_size() / 512.0}')

    optimizer = create_optimizer(
        args, model2optimize, filter_requires_grad=False)    

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Distillation
    if args.phase == 'm2mKD':
        assert args.distillation_type != 'none'
        meta_model = create_model(args.meta, 
                                  divided_depths=args.meta_divided_depths,
                                  num_classes=args.nb_classes,)
        load_meta(args, meta_model, model_is_meta=True)
        for param in meta_model.parameters():
            param.requires_grad = False
        meta_model.to(device)
        teacher = get_teacher(args)
        codes = None
        states = None
        if args.model == 'nac':
            codes = torch.randn(1, student_cfg['code_dim']).to(device) # The dimension 1 will broadcast
            if args.idx == 0 or args.idx == -1:
                states = torch.randn(args.batch_size, 1, student_cfg['state_dim']).to(device)
        criterion = m2mKDLoss(
            device, criterion, meta_model, teacher, model, args.model, args.idx, codes, states, 
            args.distillation_type, args.alpha_teacher, args.distillation_tau
        )
    else:
        assert args.distillation_type == 'none'
        teacher_model = None
        criterion = DistillationLoss(
            criterion, teacher_model, args.distillation_type, args.alpha_teacher, args.distillation_tau
        )

    utils.auto_load_model(args, model_without_ddp, optimizer, loss_scaler, lr_scheduler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device,
                              KDcriterion=criterion if args.phase == 'm2mKD' else None)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            use_amp=args.use_amp
        )

        lr_scheduler.step(epoch)
        utils.save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, lr_scheduler)

        test_stats = evaluate(data_loader_val, model, device, 
                              m2mKDcriterion=criterion if args.phase == 'm2mKD' else None)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]

        print(f'Max accuracy: {max_accuracy:.2f}%')

        used_time = time.time() - start_time
        used_time_str = str(datetime.timedelta(seconds=int(used_time)))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'used_time': used_time_str,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with Path(f"{args.output_dir}/log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        print('Training time {}'.format(total_time_str))
        utils.save_model(args, args.epochs-1, model_without_ddp, optimizer, loss_scaler, lr_scheduler,
                         finished=True)
        if args.save_modules:
            save_modules(args, model_without_ddp)


def model_assemb(args, model):
    assert len(model.stages) == len(args.incubation_models), f'Please provide {len(model.stages)} target modules for assembling.'
    for i, module in enumerate(model.stages):
        ckpt = torch.load(args.incubation_models[i], map_location='cpu')
        if len(ckpt['model_stages']) == 1:
            # vmoe_student
            incubated_module = ckpt['model_stages'][0]
        else:
            incubated_module = ckpt['model_stages'][i]
        module.load_state_dict(incubated_module, strict=False)


def load_meta(args, model, model_is_meta):
    """
    This function is originally module_incubation().
    """
    meta_ckpt = torch.load(args.meta_ckpt_path, map_location='cpu')
    meta_modules = meta_ckpt['model_stages']
    # loading meta modules into the hybrid model
    for i, module in enumerate(model.stages):
        if model_is_meta:
            module.load_state_dict(meta_modules[i], strict=False)
        elif i != args.idx:
            module.load_state_dict(meta_modules[i])
        module.to(torch.device(args.device))


def get_nac_student_type(idx):
    if idx == 0:
        return 'readin'
    elif idx == -1:
        return 'readout'
    return 'mediator'


def get_teacher(args):
    ckpt = torch.load(args.teacher_ckpt_path, map_location='cpu')
    root_model = create_model(ckpt['model_name'], divided_depths=ckpt['divided_depths'], num_classes=args.nb_classes)
    root_model.stages[ckpt['block_index']].load_state_dict(ckpt['block_state_dict'])
    # TODO: remove the unused parameters.
    root_model.to(torch.device(args.device))
    root_model.eval()
    start = 0
    for i, depth in enumerate(ckpt['divided_depths']):
        if i == ckpt['block_index']:
            end = start + depth
            break
        start += depth
    return (ckpt['block_index'], start, end, root_model)
        

def save_modules(args, model):
    for i, module in enumerate(model.stages):
        save_path = os.path.join(args.output_dir, f'{args.model}_{i}.pth')
        torch.save({
            'model_name': args.model,
            'block_index': i,
            'block_state_dict':module.state_dict(),
            'divided_depths': args.divided_depths,
        }, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
