import os
import tqdm
import yaml
from PIL import Image
from pathlib import Path

from typing import Optional, Callable

import yaml
import asset

import numpy as np
from torchvision.datasets import ImageFolder


WNIDS = yaml.load(
    asset.load("neural_compilers:data/assets/wnids.yml").read(), Loader=yaml.SafeLoader
)
in_dir = '../../imagenet-r'
out_dir = '../../tiny-imagenet-r'
tiny_imagenet_r_wnids = {'n02948072', 'n03424325', 'n07749582', 'n07734744', 'n02206856', 'n02909870', 'n02085620', 'n02123045', 'n02883205', 'n04275548', 'n07873807', 'n02165456', 'n02099601', 'n04146614', 'n02769748', 'n02236044', 'n02056570', 'n02423022', 'n12267677', 'n01774750', 'n02486410', 'n02410509', 'n04133789', 'n02113799', 'n01784675', 'n02268443', 'n02480495', 'n02841315', 'n07753592', 'n02129165', 'n02802426', 'n01983481', 'n01443537', 'n07720875', 'n02814860', 'n02279972', 'n04465501', 'n03649909', 'n02950826', 'n02808440', 'n07695742', 'n02395406', 'n04118538', 'n01882714', 'n02481823', 'n01770393', 'n02106662', 'n07614500', 'n01944390', 'n02793495', 'n02233338', 'n02094433', 'n02906734', 'n02843684', 'n07920052', 'n02190166', 'n01910747', 'n02226429', 'n02364673', 'n01855672', 'n07768694', 'n02099712'}
# yaml.dump(list(overlap), open("../../data/assets/tiny-imnet-r-wnids.yml", "w"))

import csv
with open("tiny-imagenet-200/wnids.txt", "r") as f:
    reader = csv.reader(f)
    tiny_imagenet_200_wnids = []
    for x in reader:
        tiny_imagenet_200_wnids.extend(x)
WNIDS["tiny-imagenet-200"] = list(tiny_imagenet_200_wnids)
WNIDS["tiny-imagenet-r"] = list(tiny_imagenet_r_wnids)
yaml.dump(WNIDS, open("data/assets/wnids.yml", "w"))

for folder in tqdm.tqdm(os.listdir(in_dir)):
    if folder not in overlap:
        continue
    subdir = os.path.join(in_dir, folder)
    for filename in os.listdir(subdir):
        in_filepath = os.path.join(subdir, filename)
        image = Image.open(in_filepath)
        new_image = image.resize((64, 64))
        outdir = os.path.join(out_dir, folder)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        outpath = os.path.join(outdir, filename)
        new_image.save(outpath)
