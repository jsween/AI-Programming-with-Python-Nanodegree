#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jonathan Sweeney
# DATE CREATED: 20230810
# REVISED DATE: 
# PURPOSE: Predicts flower name from an image along with supplying the 
#          probability of that name.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py 
#             --path <directory with the image>  *required
#             --checkpoint <path to checkpoint>
#             --top_k <top K most likely classes>
#             --category_names <use a mapping of categories to real names>
#             --gpu <use if gpu should be used>
#   Example call:
#    python predict.py input checkpoint --gpu
##

from time import time

from classifier import display_prediction
from get_data import get_predict_input_args
from utility import calc_elapsed_time, read_cat_to_name
from validation_functions import check_predict_cl_args


def main():
    start_time = time()

    in_arg = get_predict_input_args()
    check_predict_cl_args(in_arg)
    cat_to_name = read_cat_to_name(in_arg.category_names)
    display_prediction(in_arg, in_arg.checkpoint, cat_to_name)
    end_time = time()
    calc_elapsed_time(end_time - start_time)


if __name__=='__main__':
    main()