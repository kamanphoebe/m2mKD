# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
from PIL import Image
import random

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from neural_compilers.data.nlvr import (
    NLVRRawNoConcat,
    NLVRPreprocessed,
    NLVRPreprocessedPatches,
)
from neural_compilers.data.imagenet_r import ImageNetR, TinyImageNetR
from neural_compilers.data.cifar_fs import CIFAR100FS
from neural_compilers.data.cub2011 import Cub2011FS
from neural_compilers.model.external.randaugment import RandomAugment


class INatDataset(ImageFolder):
    def __init__(
        self,
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category="name",
        loader=default_loader,
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, "categories.json")) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter["annotations"]:
            king = []
            king.append(data_catg[int(elem["category_id"])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data["images"]:
            cut = elem["file_name"].split("/")
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args, use_transform=True):
    if use_transform:
        transform = build_transform(is_train, args)
    else:
        transform = None

    if args.data_set == "CIFAR100-FS":
        if args.seed:
            random.seed(args.seed)
        chosen_class = random.sample(range(100), args.k_way)
        n_sample = args.n_shot if is_train else args.val_sample
        dataset = CIFAR100FS(args.data_path, chosen_class=chosen_class, n_sample=n_sample, 
                        train=is_train, transform=transform, seed=args.seed)
        nb_classes = args.k_way
    elif args.data_set == "CUB-FS":
        if args.seed:
            random.seed(args.seed)
        chosen_class = random.sample(range(200), args.k_way)
        n_sample = args.n_shot if is_train else args.val_sample
        dataset = Cub2011FS(args.data_path, chosen_class=chosen_class, n_sample=n_sample, 
                        train=is_train, transform=transform, seed=args.seed)
        nb_classes = args.k_way
    elif args.data_set == "CIFAR":
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == "TINY-IMNET":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200
    elif args.data_set == "IMNET":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        if getattr(args, "cache_data_path", None) is None:
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            from neural_compilers.data.cached_index_image_folder import (
                CachedIndexImageFolder,
            )

            split = "train" if is_train else "val"
            # For spectators: building a list of images takes more than a few minutes,
            # and this cache is a way of avoiding that.
            cache_path = os.path.join(args.cache_data_path, f"inet_cache_{split}.pkl")
            assert os.path.exists(cache_path), f"{cache_path} does not exist!"
            dataset = CachedIndexImageFolder(root, cache_path, transform=transform)
        nb_classes = 1000
    elif args.data_set == "INAT":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2018,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "INAT19":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2019,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "IMNET-R":
        assert not is_train
        dataset = ImageNetR(args.data_path, transform=transform)
        # INetR classes are mapped to INet classes
        nb_classes = 1000
    elif args.data_set == "NLVR":
        dataset = NLVRRawNoConcat(args.data_path, train=is_train, transform=transform)
        nb_classes = 2
    elif args.data_set == "TINY-IMNET-R":
        assert not is_train
        dataset = TinyImageNetR(args.data_path, transform=transform)
        # Tiny-INetR classes are mapped to Tiny-INet classes
        nb_classes = 200
    else:
        raise NotImplementedError

    return dataset, nb_classes


def build_transform(is_train, args):
    if args.data_set != "NLVR":
        return _build_imagenet_transforms(is_train, args)
    else:
        return _build_nlvr_transforms(is_train, args)


def _build_nlvr_transforms(is_train, args):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    if is_train and args.use_augmentation:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    args.input_size, scale=(0.5, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Equalize",
                        "Brightness",
                        "Sharpness",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (args.input_size, args.input_size), interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transform


def _build_imagenet_transforms(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
