#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jonathan Sweeney
# DATE CREATED: 20230810
# REVISED DATE: 
# PURPOSE: Utility functions 
#
##
import json 

import torch
import numpy as np
from collections import OrderedDict
from PIL import Image
from torch import nn
from torchvision import models

def calc_elapsed_time(tot_time):
    print("\n*** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )


def load_checkpoint(filepath):
    model_info = torch.load(filepath)
    model = model_info['transfer_model']
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.load_state_dict(model_info['state_dict'])

    return model, model_info


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image)
    width, height = im.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if (max_element == 0):
        min_element = 1
    else:
        min_element = 0
    aspect_ratio=picture_coords[max_element]/picture_coords[min_element]
    new_picture_coords = [0,0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)   
    width, height = new_picture_coords
    left = (width - 244)/2
    top = (height - 244)/2
    right = (width + 244)/2
    bottom = (height + 244)/2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image = np_image / [255,255,255]
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def read_cat_to_name(path='cat_to_name.json'):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name