#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jon Sweeney
# DATE CREATED: 20230809
# REVISED DATE:
# PURPOSE:  This set of functions to fetch various data
#
##
import argparse

import json
import torch
from torchvision import datasets, transforms


def get_train_input_args():
    """
    Retrieves and parses 1 to 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these command line arguments. If
    the user fails to provide all of the arguments, then the default values are
    used for the missing arguments.
    Command Line Arguments:
      1. dir : image data directory
      2. save : directory to save checkpoints
      3. arch : Pytorch model
      4. learning_rate : learning rate
      5. hidden_units : number of hidden units
      6. epochs : number of epochs
      7. gpu : use GPU for training
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # TODO: Finish getting all arguments 
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # TODO: Remove the double hyphen
        "--dir",
        type=str,
        help="Required: The path to the directory containing flower images (e.g. flowers/)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="check.pth",
        help="The path to the directory to save the checkpoints",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg",
        help="The CNN Model Architecture to use (e.g. vgg)",
    )
    print("Command line arguments parsed")

    return parser.parse_args()


def get_predict_input_args():
    print('Parsing command line arguments...')
    parser = argparse.ArgumentParser(description="Parser for prediction command line arguments")
    parser.add_argument(
        "image_path",
        type=str,
        help="Required: The path to the directory containing flower image (e.g. flowers/valid/1/12345.jpg)",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Required: The file containing the classification model (e.g. check.pth)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default="5",
        help="Show the top K likely classes (e.g. 3)",
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="File to be used for category names (e.g. cat_name.json)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        help="Enable gpu",
    )
    print("Command line arguments parsed")

    return parser.parse_args()


def get_train_data():
    """
    Gets the data directories

    Returns:
      Image data organized by category
    """
    # TODO: Change to use command line arguments
    print("Getting data...")
    data_dir = "flowers"
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    return process_images([train_dir, valid_dir, test_dir])


def process_images(data_dirs):
    """
    Processes the raw images to be ready to be used to train the model

    Returns:
      JSON object with image data broken into train, valid, test and the labels
    """
    train_dir, valid_dir, test_dir = data_dirs
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # Define transforms for training, validation, and testing sets
    modified_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=modified_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=modified_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=modified_transforms)

    # Define the dataloaders
    trainloaders = torch.utils.data.DataLoader(
        train_datasets, batch_size=32, shuffle=True
    )
    validloaders = torch.utils.data.DataLoader(
        valid_datasets, batch_size=32, shuffle=True
    )
    testloaders = torch.utils.data.DataLoader(
        test_datasets, batch_size=32, shuffle=True
    )

    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)

    print("Image data loaded")
    return {
        "train_loaders": trainloaders,
        "valid_loaders": validloaders,
        "test_loaders": testloaders,
        "labels": cat_to_name,
    }
