#!/usr/bin/env python
import os

from setuptools import setup

setup(
    name="neural-compilers",
    version="1.0",
    description="Neural Compiler repository",
    author="XXXX-2, XXXX-3",
    author_email="XXXX-1",
    install_requires=[
        "numpy",
        "torch",
        "networkx",
        "wandb",
        "timm==0.4.12",
        "einops",
        "speedrun @ git+ssh://git@github.com/inferno-pytorch/speedrun@dev#egg=speedrun",
        "wormulon @ git+ssh://git@github.com/XXXX-11/wormulon@main#egg=wormulon",
    ],
    extras_require={},
    py_modules = [],
)
