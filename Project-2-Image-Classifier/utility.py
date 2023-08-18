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
from PIL import Image

def calc_elapsed_time(tot_time):
    """
    Converts and displays total elapsed time

    Parameters:
        tot_time : amount of time between start and end
    Returns:
        None
    """
    print("\n*** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )


def load_checkpoint(filepath):
    """
    Loads the last saved classifier

    Parameters:
        filepath: the path to the saved classifier
    Returns:
        the saved model
    """
    model_info = torch.load(filepath)
    model = model_info['transfer_model']
    model.load_state_dict(model_info['state_dict'])
    model.classifier = model_info['classifier']

    return model


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


def read_cat_to_name(path):
    """
    Reads the category to name mapping

    Parameters:
        path: path to file
    Returns:
        category to name mapping in json object
    """
    with open(path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name