#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jonathan Sweeney
# DATE CREATED: 20230809
# REVISED DATE:
# PURPOSE: Classifies flower images using a pretrained CNN model. 
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#      python train.py 
#             --dir <directory with images>  *required
#             --save <directory to save checkpoints>
#             --arch <model>
#             --learning_rate <learning rate>
#             --hidden_units <hidden units>
#             --epochs <number of epochs>
#             --gpu <use if gpu should be used for training
#   Example call:
#    python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
##

from time import time

from PIL import Image

import classifier as clsf
from get_data import *
from validation_functions import *

def main():
    start_time = time()
    in_arg = get_input_args()
    check_command_line_arguments(in_arg)

    data = get_data()

    classifier = clsf.build(in_arg.arch, data)
    end_time = time()

    tot_time = end_time - start_time
    print("\n*** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )


if __name__=='__main__':
    main()    
