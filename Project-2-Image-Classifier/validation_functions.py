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


def check_command_line_arguments(in_arg):
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
    if in_arg is None:
        print(
            "* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined."
        )
        return
    if in_arg.dir is None:
        raise Exception(
            "Data directory not provided: A path to the data dir is required (e.g. ./flowers/)."
        )
    elif not os.path.isdir(in_arg.dir):
        raise Exception(f"Invalid directory: '{in_arg.dir}' does not exist.")
    else:
        # prints command line agrs
        print(
            "Command Line Arguments:\n\tdir =",
            in_arg.dir,
            "\n\tarch =",
            in_arg.arch,
        )
