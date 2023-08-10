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

from classifier import *
from get_data import *
from validation_functions import *

from torch import optim

def main():
    start_time = time()
    in_arg = get_input_args()
    check_command_line_arguments(in_arg)

    data = get_data()

    model = build_classifier(in_arg.arch)
    train_classifier(model, data['train_loaders'], data['valid_loaders'], 3, 40, device='cpu')
 
    check_accuracy_on_test(data['test_loaders'], model)

    save_model(model, data, in_arg.save)
    
    end_time = time()

    tot_time = end_time - start_time
    print("\n*** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )


if __name__=='__main__':
    main()    
