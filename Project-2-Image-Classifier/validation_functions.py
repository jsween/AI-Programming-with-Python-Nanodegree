#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jon Sweeney
# DATE CREATED: 20230809
# REVISED DATE: 20230810
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
            "Data directory not provided: A path to the data dir is required (e.g. ./flowers)."
        )
    elif not os.path.isdir(in_arg.dir):
        raise Exception(f"Invalid directory: '{in_arg.dir}' does not exist.")
    sub_dirs = os.listdir(in_arg.dir)
    if "train" not in sub_dirs and "valid" not in sub_dirs and "test" not in sub_dirs:
        raise Exception(
            f"Missing one or more of sub directories train, valid, test. Found {sub_dirs}"
        )
    if in_arg.gpu and not torch.cuda.is_available():
        raise Exception("ERROR: GPU not detected on this machine. Do not select '--gpu' flag or try again")
    if in_arg.gpu and torch.cuda.is_available():
        in_arg.gpu = "cuda"
    else:
        if in_arg.gpu and not torch.cuda.is_available():
            print("WARNING: GPU is not available. Using CPU.")
        in_arg.gpu = "cpu"
    if in_arg.hidden_units is None:
        if "densenet" in in_arg.arch:
            in_arg.hidden_units = 1024
        elif "resnet" in in_arg.arch: 
            in_arg.hidden_units = 1024
        else:
            in_arg.hidden_units = 4096
    print(
        f"Command Line Arguments:\n\tdir = {in_arg.dir}\n",
        f"\tsave = {in_arg.save}\n\tarch = {in_arg.arch}\n",
        f"\tlearning_rate = {in_arg.learning_rate}\n\thidden_units = {in_arg.hidden_units}\n",
        f"\tgpu = {in_arg.gpu}"
    )


def check_accuracy_on_test(testloader, model, device):
    """
    Checks the accurracy on a set of test images

    Parameters:
        testloader: data loader of test images
        model: classifier
        device: cpu or gpu device
    Returns: 
        Percentage of correct classifications
    """
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


def check_predict_cl_args(in_arg):
    """
    Check Prediction Command Line Arguments

    Parameters:
     in_arg: command line arguments object
    """
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
