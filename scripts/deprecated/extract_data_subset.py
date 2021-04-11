import os
import numpy as np 
from Bio import SeqIO
from multiprocessing import Pool



#Data_sets
global dir_
dir_ = '/mnt/Archive/Data_Sets/OGT'
file = 'ogt.fasta'
global topt_dict
topt_dict = {'id':[], 'ogt':[], 'seq':[]}


for rec in SeqIO.parse(os.path.join(dir_,file), 'fasta'):
    topt_dict['id'].append(rec.id)
    topt_dict['ogt'].append(float(rec.description.split()[-1]))
    topt_dict['seq'].append(rec.seq)
    


temperatures=[(i,i+11) for i in range(4,103,11)]

def filter_temp(temp):
    global topt_dict
    f = open(os.path.join(dir_,'ogt_{}_{}.fasta'.format(temp[0],temp[1])),'w')
    max_count = 10000
    count= 0
    for id_, seq, ogt in zip(topt_dict['id'], topt_dict['seq'], topt_dict['ogt']):
        if ogt>=temp[0] and ogt<temp[1]:
            f.write('>{} {}\n{}\n'.format(id_,ogt,seq))
            count += 1
        else:
            continue
        if count >= max_count:
            break
            
    f.close()
    return 0

p = Pool(9)
p.map(filter_temp, temperatures)