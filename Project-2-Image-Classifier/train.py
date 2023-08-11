#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jonathan Sweeney
# DATE CREATED: 20230809
# REVISED DATE: 20230810
# PURPOSE: Trains a new network on a flower data set and save the model as
#          a checkpoint.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py
#             --dir <directory with images>  *required
#             --save <directory to save checkpoints>
#             --arch <model>
#             --learning_rate <learning rate>
#             --hidden_units <hidden units>
#             --epochs <number of epochs>
#             --gpu <use if gpu should be used>
#   Example call:
#    python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
##

from time import time

from classifier import *
from get_data import get_train_input_args, get_train_data
from validation_functions import check_train_cl_args, check_accuracy_on_test
from utility import calc_elapsed_time


def main():
    start_time = time()
    in_arg = get_train_input_args()
    check_train_cl_args(in_arg)

    data = get_train_data(in_arg.dir)
    model = build_classifier(in_arg.arch, in_arg.hidden_units)
    train_classifier(
        model, data["train_loaders"], data["valid_loaders"], in_arg.epochs, 40, in_arg.learning_rate, in_arg.gpu
    )

    check_accuracy_on_test(data["test_loaders"], model, in_arg.gpu)

    save_model(model, in_arg.save)

    end_time = time()
    calc_elapsed_time(end_time - start_time)


if __name__ == "__main__":
    main()
