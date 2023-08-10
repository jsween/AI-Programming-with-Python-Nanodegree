#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jon Sweeney
# DATE CREATED: 20230809
# REVISED DATE:
# PURPOSE:  This set of functions can be used to check arguments passed
#           into functions to verify code runs properly.
#
##

import os

import torch


def check_train_cl_args(in_arg):
    """
    Prints each of the command line arguments passed in as parameter in_arg,
    assumes you defined all three command line arguments
    Parameters:
     in_arg: command line arguments object
            dir : dir path
                image data directory
            save : dir path
                directory to save checkpoints
            arch : model
                Pytorch model
            learning_rate : float
                learning rate
            hidden_units : uint
                number of hidden units
            gpu : None
                use GPU for training
    Returns:
     Nothing - just prints to console
    """
    print("Validating command line arguments...")
    if in_arg.dir is None:
        raise Exception(
            "Data directory not provided: A path to the data dir is required (e.g. ./flowers/)."
        )
    elif not os.path.isdir(in_arg.dir):
        raise Exception(f"Invalid directory: '{in_arg.dir}' does not exist.")
    sub_dirs = os.listdir(in_arg.dir)
    if "train" not in sub_dirs and "valid" not in sub_dirs and "test" not in sub_dirs:
        raise Exception(
            f"Missing one or more of sub directories train, valid, test. Found {sub_dirs}"
        )
    else:
        # prints command line agrs
        # TODO: Print out 
        print(
            "Command Line Arguments:\n\tdir =",
            in_arg.dir,
            "\n\tarch =",
            in_arg.arch,
        )


def check_predict_cl_args(in_arg):
    print("Validing CL arguments...")
    if in_arg.image_path is None:
        raise Exception(
            "Image Path not provided. A path to the file to predict is required."
        )
    if in_arg.checkpoint is None:
        raise Exception(
            "Image Path not provided. A path to the file to predict is required."
        )
    print("Args are valid.")


def check_accuracy_on_test(testloader, model, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        "Accuracy of the network on the 10,000 test images: %d %%"
        % (100 * correct / total)
    )
    return correct / total
