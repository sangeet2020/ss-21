#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
<Function of script>
"""

import os
import sys
import pdb
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict

def data_preparation(text, args, test_size:float=0.2):
    """ pre-process text corpus, split into train-test and save"""
    tokens_list = text.split('\n')
    tokens_list = [s.strip() for s in tokens_list]
    k = int(len(text) * (1 - test_size))
    with open(args.out_dir + '/train_beng.txt', 'w') as train_fpw, open(args.out_dir + '/test_beng.txt', 'w') as test_fpw:
        train_fpw.write(text[:k])
        test_fpw.write(text[k:])


def model_train(args):
    # Step 1: train the model and specify the target vocabulary size
    cmd_step1 = "spm_train \
                --input=" + str(args.out_dir + '/train_beng.txt') + " \
                --model_prefix=" + str(args.model_dir + "/s1") + " \
                --vocab_size=" + str(args.vocab_size) + "     \
                --character_coverage=1.0 \
                --model_type=bpe"
    
    # Step 2: segment the original text using this model.
    cmd_step2 = "spm_encode \
                --model=" + str(args.model_dir + "/s1.model") + " \
                --output_format=piece \
                < " + str(args.out_dir + '/train_beng.txt') + " \
                > " + str(args.out_dir + '/segmented.txt')
    
    cmd_step3 = "spm_decode \
                --model=" + str(args.model_dir + "/s1.model") + " \
                --input_format=piece \
                < " + str(args.out_dir + '/segmented.txt')+ " \
                > " + str(args.out_dir + '/original.txt')
                
    
    os.system(cmd_step1)
    os.system(cmd_step2)
    os.system(cmd_step3)

def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    data_preparation(Path(args.corpus_fp).open('r').read(), args)
    model_train(args)
    # pdb.set_trace()

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_fp", help="path to text copus file")
    parser.add_argument("out_dir", help="path to save output files") 
    parser.add_argument("model_dir", help="path to save trained models") 
    parser.add_argument("-vocab_size", default=1000, type=int, help='vocab size')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()