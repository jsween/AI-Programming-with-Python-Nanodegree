#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jon Sweeney
# DATE CREATED: 20230809
# REVISED DATE:
# PURPOSE:  This set of functions to interact with the classifier object
#
##

from collections import OrderedDict
from torch import nn
from torchvision import models


def build(arch, data):
    """
    Builds the classifier model object
    """
    if arch == "vgg":
        model = models.vgg19(weights=True)
    else:
        print(f"WARNING: {arch} is unsupported. Defaulting to vgg model.")
        model = models.vgg19(weights=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(25088, 4096)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(4096, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier

    return model
