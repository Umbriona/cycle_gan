import os

import numpy as np


import argparse


parser = argparse.Argumentparser(""" """)

parser.add_argmunet('-i', '--input', type=str, required=True,
                    help="")
parser.add_argmunet('-o', '--output', type=str, required=True,
                    help="")
parser.add_argmunet('-l', '--max_length', type=int, default=512,
                    help="")
parser.add_argmunet('-i', '--input', type=str, required=True,
                    help="")
parser.add_argmunet('-i', '--input', type=str, required=True,
                    help="")

parser.add_argmunet('-i', '--input', type=str, required=True,
                    help="")

# Define
aas = 'ACDEFGHIKLMNPQRSTVWYX'

def get_ngrams(seq, word_length):
    
    for ofset in range(word_length):
        list_of_ngrams = [ str(dat_thermo['seq'][k][i+ofset:i+word_length+ofset]) for i in range(0,len(dat_thermo['seq'][k])-word_length-ofset,word_length)]
    for key in list_of_ngrams:
        features_thermo[k,dict_index[key]]+=1
    length = len(str(dat_thermo['seq'][k]))
    features_thermo[k,:]/=length

def read_data(file, max_length = 512):
    data = {'id':[], 'ogt':[], 'seq':[]}
    count = 0
    for rec in SeqIO.parse(file, 'fasta'):
        if len(str(rec.seq)) > max_length:
            continue
        count += 1
        dat_thermo['id'].append(rec.id)
        dat_thermo['ogt'].append(float(rec.description.split()[-1]))
        dat_thermo['seq'].append(rec.seq)

def main(args):
    
    return 0

if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)