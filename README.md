# m2mKD

This repository contains the source code for [m2mKD: Module-to-Module Knowledge Distillation for Modular Transformers]().

## Overview

We propose a general module-to-module knowledge distillation (m2mKD) method for transferring knowledge between modules. Our approach involves teacher modules split from a pretrained monolithic model, and student modules of a modular model. m2mKD separately combines these modules with a shared meta model and encourages the student module to mimic the behavior of the teacher module. By applying m2mKD to [NAC](https://proceedings.neurips.cc/paper_files/paper/2022/file/32f227c41a0b4e36f65bebb4aeda94a2-Paper-Conference.pdf) and [V-MoE](https://proceedings.neurips.cc/paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf) models, we achieve improvements in both IID accuracy and OOD robustness.

![pipeline](./images/pipeline.png)

## Usage

Our experiments are conducted on NAC and V-MoE models. The instructions for training these models can be found in [TRAIN-NAC.md](./TRAIN-NAC.md) and [TRAIN-MOE.md](./TRAIN-MOE.md), respectively. If you would like to have a quick try about m2mKD, we have provided the checkpoints of NAC student modules for Tiny-ImageNet. See [TRAIN-NAC.md](./TRAIN-NAC.md) for details.

## Acknowledgement

Our implementation is mainly based on [Deep-Incubation](https://github.com/LeapLabTHU/Deep-Incubation). 

## Citation

If you use the code, please cite our paper:
```
```