import os
import tarfile
import math
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse

# Defines
FASTA_STRING = ">{} {}\n{}\n"
TMP_DIR = "/tmp/tmp_img"

parser = argparse.ArgumentParser(""" """)

parser.add_argument("-i", "--input", type = str, required=True)
parser.add_argument("-m", "meta_data", type = str, default = None)
parser.add_argument("-o", "--output", type = str, required=True)

parser.add_argument('-v', '--verbose', type=int, choices=[0,1,2], default = 1)

def extract_all(args):
    
    if os.isdir(TMP_DIR):
        
    
    with tarfile.open(args.input, 'r') as file_reader:
    file_reader.extractall(TMP_DIR)
    
    base_name = os.path.basename(args.input).split('.')[0]
    base_dir = os.path.basedir(args.input)
    directory = os.path.join(base_dir, base_name)
    return directory

def get_meta(df_meta, genome):
    genome_id = int(genome.split('.')[0])

    # get meta

    row = df_meta.loc[df_meta["IMG Genome ID "] == genome_id]
    sample_temp = row["Sample Collection Temperature"]

    try:
        nan = math.isnan(sample_temp.values[0])
        if nan:
            val = '-'
    except TypeError:
        val = sample_temp.values[0].split(" ")[0]
        unit = sample_temp.values[0].split(" ")[-1]
        if unit != "C":
            raise TypeError("Conversion from {} not implemented needs to be C".format(unit))
    except:
        raise ValueError("Value of type {} for Sample Collection Temp is not permitted".format(type(sample_temp.values[0]))) 
    sample_temp = val
    range_temp  = row["Temperature Range"].values[0]
    return genome_id, range_temp, sample_temp 
        
def get_seqs(f, genome_id, range_temp, sample_temp):
    # get seqs
    prot_fasta_file = [info.name for info in f.getmembers() if info.name[-3:] == "faa"]
    prot_fasta_file_obj = f.extractfile(prot_fasta_file[0])

    prot_fasta_file = StringIO("".join([line.decode('utf-8') for line in prot_fasta_file_obj.readlines()]))

    #collect data

    fasta_dict = {"Id": [], "records": [], "seq": []}
    for rec in SeqIO.parse(prot_fasta_file,"fasta"):
        fasta_dict["Id"].append(rec.id)
        fasta_dict["records"].append(rec.description + "| {} | {} | {}".format(genome_id, range_temp , sample_temp))
        fasta_dict["seq"].append(str(rec.seq))
    return fasta_dict
    

def main(args):
    
    directory = extract_all(args)
    df_meta = pd.read_csv(args.meta_data, sep='\t')
    
    for genome in os.listdir(directory):
        if genome[-7:] == ".tar.gz" :
            # Get meta data
            genome_id, range_temp, sample_temp = get_meta(df_meta, genome)
            # Get Sequence data
            with tarfile.open(os.path.join(data_dir,genome), 'r') as f:
                fasta_dict = get_seqs(f, genome_id, range_temp, sample_temp)
            # Write fasta 
            n_seqs = len(fasta_dict["Id"])
            temperature = fasta_dict["records"][0].split(" | ")[-1]
            with open(args.output, "a") as d:
                for seq_id, seq_rec, seq in zip(fasta_dict["Id"], fasta_dict["records"], fasta_dict["seq"]):
                    d.write(FASTA_STRING.format(seq_id,seq_rec,seq))
            print("Done intergrating {} sequences with {} sample temp from file {} ".format(n_seqs, temperature, genome))
        else:
            print("skipping file {}".format(genome))
    
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
