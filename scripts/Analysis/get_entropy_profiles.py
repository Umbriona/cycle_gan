#! /usr/bin/env python

import numpy as np
import math
import json
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type=str, default = "../../data/OG_ali/id.ali")
parser.add_argument('-o', '--output', type=str, default = "../../data/OG_entropy/id.csv")

AAS = 'ACDEFGHIKLMNPQRSTVWYX'

def import_alignment(name):
    dict_alignment_all    = {'id':[], 'ogt':[], 'seq':[]}
    dict_alignment_meso   = {'id':[], 'ogt':[], 'seq':[]}
    dict_alignment_thermo = {'id':[], 'ogt':[], 'seq':[]}

    for rec in SeqIO.parse(name, 'fasta'):
        dict_alignment_all['id'].append(rec.id)
        dict_alignment_all['ogt'].append(float(rec.description.split()[-1]))
        dict_alignment_all['seq'].append(str(rec.seq))

        if float(rec.description.split()[-1]) <= 20:
            dict_alignment_meso['id'].append(rec.id)
            dict_alignment_meso['ogt'].append(float(rec.description.split()[-1]))
            dict_alignment_meso['seq'].append(str(rec.seq))
        elif float(rec.description.split()[-1]) <= 70:
            dict_alignment_thermo['id'].append(rec.id)
            dict_alignment_thermo['ogt'].append(float(rec.description.split()[-1]))
            dict_alignment_thermo['seq'].append(str(rec.seq))
    dict_ ={'All':dict_alignment_all, 'Meso':dict_alignment_meso, 'Thermo': dict_alignment_thermo}
    return dict_

def index_aa(seq):
    idx = []
    for i, aa in enumerate(seq):
        if aa is not "-":
            idx.append(i)
    return idx

def aa_at_index(idx, ali):
    list_mutations=[]
    for j in idx:
        tmp=[]
        for i, seq in enumerate(ali['seq']):
            tmp.append(seq[j])
        list_mutations.append(tmp)
    return list_mutations

def prob_mutation(seq, idx, msa_aa):
    total_counts_per_residue = []
    insertion_site = np.zeros((len(idx),))
    for index, res_aa in enumerate(msa_aa):
        count_residue = 1
        for aa in res_aa:
            if aa is not '-':
                count_residue += 1
        total_counts_per_residue.append(count_residue)
        if all(np.array(res_aa) == '-'):
            insertion_site[index] = 1
    
    same_counts_per_residue = []
    for i, res_aa in enumerate(msa_aa):
        count_residue = 1
        for aa in res_aa:
            if aa is seq[idx[i]]:
                count_residue += 1
        same_counts_per_residue.append(count_residue)
   
    return np.array(same_counts_per_residue)/np.array(total_counts_per_residue), insertion_site

def residy_entropy(seq, idx, msa_aa):
    total_counts_per_residue = []
    insertion_site = np.zeros((len(idx),))
    for index, res_aa in enumerate(msa_aa):
        count_residue = 1
        for aa in res_aa:
            if aa != '-':
                count_residue += 1
        total_counts_per_residue.append(count_residue)
        if all(np.array(res_aa) == '-'):
            insertion_site[index] = 1
    
    entropy_per_residue = []
    for i, res_aa in enumerate(msa_aa):
        counts_aa = {aa:0 for aa in AAS}
        for aa in res_aa:
            if aa is not '-':
                counts_aa[aa] += 1
        counts_aa[seq[idx[i]]] += 1
        entropy=0
        for key in counts_aa.keys():
            tmp = counts_aa[key] / total_counts_per_residue[i] 
            if tmp != 0:
                entropy -= tmp * math.log(tmp, 2)
        entropy_per_residue.append(entropy)
    return entropy_per_residue, insertion_site


    
    
def main(args):
    

    dicts = import_alignment(args.input)
    

    profile_dict = {}
    for idx, id_ in enumerate(dicts['All']['id']):
        seq = dicts['All']['seq'][idx]
        ogt = dicts['All']['ogt'][idx]
        index = index_aa(seq)
        msa_aa = aa_at_index(index, dicts['All'])
        entropy_per_residue, insertions = residy_entropy(seq, index, msa_aa)
        prob_per_residue, insertions = prob_mutation(seq, index, msa_aa)
        profile_dict[id_] = {'seq': seq,
                             'ogt': ogt,
                             'entropy': list(entropy_per_residue),
                             'prob': list(prob_per_residue),
                             'insertion': list(insertions)}
    
    
    with open(args.output, 'w') as f:
        json.dump(profile_dict, f)
    
    return 0

if(__name__ == "__main__"):
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    