import os

import pandas as pd
import numpy as np
import json

from Bio import SeqIO

import argparse


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type=str, required=True, 
                   help="Annotations file from Eggnog")

parser.add_argument('-o', '--output', type=str, required=True,
                   help="output json file, if fasta files are written those files will end up in the same directory")

parser.add_argument('--write_fasta', action="store_true",
                   help=" Weather or not to write fasta files of ogs")
parser.add_argument('-f', '--fasta', type=str, default= None,
                    help="input fasta file")

parser.add_argument('--low_boundary', type=int, default=100)
parser.add_argument('--high_boundary', type=int, default=100000)

def count_ogs(df_annot):
    og_count = {}
    og_id    = {}
    for row in df_annot.iterrows():
        if row[1]['eggNOG_OGs'] != '-': # only return annotated queries
            string = row[1]['eggNOG_OGs']
            query  = row[1]['#query']
            lis = string.split(',')
            for OG in lis[1:] if len(lis) > 1 else lis: # do not include root
                og = OG.split('@')[0]
                if og not in og_count.keys():
                    og_count[og] = 1
                    og_id[og] = [query]               
                else:
                    og_count[og] += 1
                    og_id[og].append(query)
                    
    return og_count, og_id

def write_list_og(og_count, args):
    print("writing list of OGs with counts")
    with open(args.output, 'w') as f:
        json.dump(og_count, f)

def write_fasta(list_query, og_name, fasta_dict, args):
    """
    Thus function will write a fasta file with all entries that is
    in both list_query and in original fasta file args.fasta
    """

    file_og_fasta = os.path.join(os.path.dirname(args.output),"{}.fasta".format(og_name))
    count=0
    with open(file_og_fasta, "w") as file_writer:
        for i, rec in enumerate(list_query):
            try:
                file_writer.write(">{} {}\n{}\n".format(rec, fasta_dict[str(rec)][-1], fasta_dict[str(rec)][0]))
                count += 1
            except:
                raise KeyError('Key {} of type {} not present in fasta file'.format(rec, type(rec)))
                
    print("Written fasta file {file} with {num} sequences".format(file=args.fasta, num=count))
    return 0

def read_data(file):
    try:
        df_annotations = pd.read_csv(file, sep = '\t')
    except:
        df_annotations = pd.read_csv(file, sep = '\t', skiprows=4, skipfooter=3)
    return df_annotations

def main(args):
    
    ## Read fasta
    df_annotation = read_data(args.input)
    print("Loaded annotations")
    ## Get OG and counts
    og_count, og_id = count_ogs(df_annotation)
    print("Done counting OGs")
    ## writing list of OGs 
    write_list_og(og_count, args)
    print("Done writing list og OGs")
    ## Writing fasta files of OGs 
    if args.write_fasta:
        ## load fasta to dict
        fasta_dict = {}
        for i, rec in enumerate(SeqIO.parse(args.fasta, 'fasta')):
            fasta_dict[rec.id] = (rec.seq, rec.description.split()[-1])
            
        for og_name, list_query in og_id.items():
            if og_count[og_name] >= args.low_boundary and og_count[og_name] < args.high_boundary:
                print('og count', og_count[og_name])
                write_fasta(list_query, og_name, fasta_dict,  args)
        print("Done writing OG fastas")
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    print("Parsed args")
    if args.write_fasta: assert args.fasta != None, "If you want construct fasta files of OGs you need to provide a fasta file with sequences"
    main(args)