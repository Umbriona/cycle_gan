#! /usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type=str, default="../../data/annot/NOG_annotations.tsv.emapper.annotations")
parser.add_argument('-o', '--output_dir', type = str, default="../../data/OG_Classes")
parser.add_argument('--min_num', type = int, default = 50, 
                   help = "minimum number of sequences in a OG to be written")

def read_annot(args):
    file_readme = os.path.join(os.path.dirname(args.input),'README')
    with open(file_readme, 'r') as f:
        str_= f.read()
        list_header_footer = [ int(i)-1 for i in str_.split('\n')[:-1]]

    df = pd.read_csv(args.input, sep='\t',skiprows=list_header_footer)
    return df

def og_broad(df):
    df['broad_OG_name'] = [ i.split(',')[0] for i in df['eggNOG_OGs']]
    return df

def ogt_seq(df, args):

    dict_fasta_ogt = {}
    dict_fasta_seq = {}
    temp_dir = os.path.dirname(os.path.dirname(args.input))
    fasta_file = os.path.join(temp_dir, 'OGT_Classes', "ogt_all.fasta")
    for rec in SeqIO.parse(fasta_file, 'fasta'):
        dict_fasta_ogt[rec.id] = rec.description.split()[-1]
        dict_fasta_seq[rec.id] = str(rec.seq)
                               
    temperatures=[]
    sequences = []
    for id_ in df['#query']:
        temperatures.append(dict_fasta_ogt[id_])
        sequences.append(dict_fasta_seq[id_]) 
    df['OGT'] = temperatures
    df['seq'] = sequences
    return df

def get_OG_clusters(df, args):
    path = args.output_dir
    for i, key in enumerate(df['broad_OG_name'].unique()):
        if sum(df['broad_OG_name']== key) >= args.min_num:
            df_tmp = df[df['broad_OG_name']== key]
            with open(os.path.join(path, key.replace('|','-')+'_'+ str(len(df_tmp))+'.fasta'), 'w') as f:
                for index, row in df_tmp.iterrows():
                    f.write(">{} {}\n{}\n".format(row['#query'], row['OGT'], row['seq']))

    return 0

def main(args):
    os.mkdir(args.output_dir)
    df = read_annot(args)
    df = og_broad(df)
    df = ogt_seq(df, args)
    get_OG_clusters(df, args)
    
    return 0
                               
if(__name__ == "__main__"):
    args = parser.parse_args()
    main(args)