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
import re
import argparse
import string
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split

# 1 Data Preparation
def remove_emoticons(text):
    emoji_pattern = re.compile(
        u'([\U0001F1E6-\U0001F1FF]{2})|' # flags
        u'([\U0001F600-\U0001F64F])'     # emoticons
        "+", flags=re.UNICODE)

    return emoji_pattern.sub('', text)
    
    
# def data_preparation(text, args, test_size:float=0.2):
#     """ pre-process text corpus, split into train-test and save"""
#     # Remove all punctuations except "? , !"
#     remove = string.punctuation
#     remove = remove.replace("?", "").replace(",", "").replace("!", "")
#     pattern = r"[{}]".format(remove) # create the pattern
#     text = re.sub(pattern, "", text) 
#     # text = text.translate(str.maketrans('', '', string.punctuation))

#     all_lines = text.split('\n')
#     # Strip emoticons, english strings, whitespaces
#     prepro_text = []
#     for line in all_lines:
#         if line:
#             line = remove_emoticons(line) # remove flag symbols
#             line = re.sub("[A-Za-z]+","",line) # remove english chars
#             line = re.sub(r"\s+", " ", line) # remove white spaces
#             line = re.sub(r'(\W)(?=\1)', '', line)
#             line = line.replace("?", "?\n").replace(",", ",\n").replace("!", "!\n").replace("।", "।\n")
#             lines = line.split("\n")
#             for line in lines:
#                 prepro_text.append(line.strip())
            
#     with open(args.out_dir + '/original_dataset.txt', 'w') as fpw:
#         fpw.write('\n'.join(prepro_text))

#     # k = int(len(prepro_text) * (1 - test_size))
#         train, test = train_test_split(prepro_text, shuffle=False, random_state=123)
#     with open(args.out_dir + '/train_bn.txt', 'w') as train_fpw, open(args.out_dir + '/test_bn.txt', 'w') as test_fpw:
#         train_fpw.write('\n'.join(train))
#         test_fpw.write('\n'.join(test))
        
def data_preparation(text, args, test_size:float=0.2):
    """ pre-process text corpus, split into train-test and save"""
    all_lines = text.split('\n')
    # Strip emoticons, english strings, whitespaces
    prepro_text = []
    for line in all_lines:
        if line:
            line = remove_emoticons(line) # remove flag symbols
            line = re.sub("[A-Za-z]+","",line) # remove english chars
            line = re.sub(r"\s+", " ", line) # remove white spaces
            prepro_text.append(line.strip())
            
    with open(args.out_dir + '/original_dataset.txt', 'w') as fpw:
        fpw.write('\n'.join(prepro_text))

    # k = int(len(prepro_text) * (1 - test_size))
    train, test = train_test_split(prepro_text, shuffle=True, random_state=12)
    with open(args.out_dir + '/train_bn.txt', 'w') as train_fpw, open(args.out_dir + '/test_bn.txt', 'w') as test_fpw:
        train_fpw.write('\n'.join(train))
        test_fpw.write('\n'.join(test))
        
    
# 2  Creating data for LM
def model_train(vocab_size, prefix, args):
    # Step 1: train the model and specify the target vocabulary size
    input = str(args.out_dir + '/train_bn.txt')
    model_prefix = str(args.model_dir + '/' + prefix)
    cmd_step1 = "spm_train \
                --input=" +  input + " \
                --model_prefix=" + model_prefix + " \
                --vocab_size=" + str(vocab_size) + "     \
                --character_coverage=0.995 \
                --model_type=bpe"
    
    # Step 2: segment the original text using this model.
    model = model_prefix + '.model'
    output = str(args.out_dir + '/'+ prefix + '.txt')
    cmd_step2 = "spm_encode \
                --model=" + model + " \
                --output_format=piece \
                < " + input + " \
                > " + output
    
    # # Step 3: go from subword unit segmentation back to the original text 
    # cmd_step3 = "spm_decode \
    #             --model=" + str(args.model_dir + "/s1.model") + " \
    #             --input_format=piece \
    #             < " + str(args.out_dir + '/segmented.txt')+ " \
    #             > " + str(args.out_dir + '/original.txt')
                
    os.system(cmd_step1)
    os.system(cmd_step2)
    # os.system(cmd_step3)

# 3 LM Training
def lm_train(prefix, class_size, args):
    # train language model based on the different subword 
    # granularity: char, subword unit(small vocab), subword unit(large vocab)
    print("*"*20 + " Begin LM training " + "*"*20)
    os.makedirs('rnn_model', exist_ok=True)
    train = str(args.out_dir + '/'+ prefix + '.txt')
    valid = str(args.out_dir + '/test_bn.txt')
    hid = str(args.hid)
    bptt = str(args.bptt)
    ext = prefix+'_hid_'+hid+"_bptt_"+bptt+"_class_"+class_size
    model =  str('rnn_model/rnnlm_'+ ext)
    cmd = "rnnlm/rnnlm \
            -train " + train + " \
            -valid " + valid + " \
            -rnnlm " + model + " \
            -hidden " + hid + " \
            -rand-seed 1 \
            -debug 2 \
            -bptt " + bptt + " \
            -class " + class_size
    
    #+ " > rnn_model/" + ext + ".log"
    
    os.system(cmd)
    print("Model saved: " + model)
    print("*"*20 + " LM training DONE " + "*"*20)

# 4. Text Generation
def text_generation(prefix, class_size, args):
    os.makedirs('bn_text', exist_ok=True)
    hid = str(args.hid)
    bptt = str(args.bptt)
    ext = prefix+'_hid_'+hid+"_bptt_"+bptt+"_class_"+class_size
    model =  str('rnn_model/rnnlm_'+ ext)
    
    spm_model = str(args.model_dir + '/' + prefix + '.model')

    
    for k in range(1, 8):
        cmd = "rnnlm/rnnlm \
            -rnnlm " + model + " \
            -gen " + str(10**k) + " \
            -debug 0 \
            >> " + str('bn_text/' + ext + '_k_' + str(k) + '.txt')
            
        cmd_decode = "spm_decode \
            --model=" + spm_model + " \
            --input_format=piece \
            < " + str('bn_text/' + ext + '_k_' + str(k) + '.txt') + " \
            > " + str('bn_text/' + ext + '_k_' + str(k) + '_decoded.txt')
        
        os.system(cmd)
        os.system(cmd_decode)
        
    print("*"*20 + " Text generation completed " + "*"*20)
    

def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    data_preparation(Path(args.corpus_fp).open('r').read(), args)
    vocab_sizes = [args.vocab_sizes]
    for prefix, vocab_size in enumerate(vocab_sizes):
        class_size = str(args.class_sizes)
        prefix = "bn_s" + str(prefix+1) + "_vocab_size_" + str(vocab_size)
        model_train(vocab_size, prefix, args)
        lm_train(prefix, class_size, args)
        # text_generation(prefix, class_size, args)
    # pdb.set_trace()

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_fp", help="path to text copus file")
    parser.add_argument("out_dir", help="path to save output files") 
    parser.add_argument("model_dir", help="path to save trained models") 
    parser.add_argument("-hid", default=40, type=int, help='hidden layer size')
    parser.add_argument("-bptt", default=3, type=int, help='num of steps to propagate error back ')
    parser.add_argument("-vocab_sizes", default=70, type=int, help='class_size')
    parser.add_argument("-class_sizes", default=70, type=int, help='class_size')
    # parser.add_argument("-class_sizes", nargs="+", default=[70, 100, 650], type=list, help='class size as list')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()