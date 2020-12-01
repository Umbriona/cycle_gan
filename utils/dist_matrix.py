import math
import pandas as pd
from itertools import combinations_with_replacement
import os
from os.path import join, abspath, basename, exists
from collections import defaultdict
import numpy as np
import argparse


threetoone = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
              'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
              'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
              'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


class PDB_Features:

    def __init__(self, output_dir=abspath('.')):
        '''
        Protein Feature Calculations is a python program for calculating protein features on PDB files.
        '''
        return None


    def parse_pdb(self, filename):
        '''
        Parse a PDB file. This saves the following data
        - Type (HEADER) (self.type)
        - EC Number (default N/A) (self.ec_number)
        - Residues per chain (self.residues)
        - Residue sequence numbers (self.res_sequence_numbers)
        - Alpha and beta coordinates per chain (self.a_coords, self.b_coords)
        - All atoms per residue per chain (self.res_atoms)
        - All atom coordinates per residue per chain (self.res_atoms_coords)
        - Total length of protein (self.length)
        '''
        #if filename[-4:] != '.pdb':
        #    raise ValueError("Only PDB files allowed")
        if filename[-4:] == 'pdb' and os.stat(filename).st_size == 0:
            raise ValueError("File is empty")
        print('Parsing', filename, '...')

        self.filename = filename
        # Save residue atoms, residue atoms coordinates, residues, residue sequence numbers, chain length,
        # alpha carbon coordinates, beta carbon coordinates
        self.res_atoms, self.res_atoms_coords = {}, {}
        self.residues = {}
        self.res_sequence_number = {}
        self.chain_length = defaultdict(int)
        self.a_coords, self.b_coords = {}, {} # (x,y,z)

        self.type = 'N/A'
        self.ec_number = 'N/A'
        self.length = 0

        # If including more features, add corresponding dictionary here
        self.exp_meth = ''
        self.feature_residues = {}
        self.feature_chains = {}
        self.feature_pdm = {}
        self.feature_relative_co = {}
        self.feature_abs_co = {}
        self.feature_rog = {}
        self.feature_surface_atoms = {}
        self.feature_phipsi_angles = {}

        # Temp arrays
        res_atoms, res_atoms_coords, res_sequence_number = [], [], []
        chain_res_atoms, chain_res_atoms_coords = [], []
        residues, a_coords, b_coords = [], [], []

        old_res = -999
        first_iter = True

        with open(filename, 'r') as pdb_file:
            for row in pdb_file:
                row = row.strip()
                
                if row[:6] == 'EXPDTA':
                    self.exp_meth = row[10:15]
                # Save the HEADER for the "type" column. For instance OXIDOREDUCTASE
                if row[:6] == 'HEADER':
                    self.type = row[6:50].strip()

                # If this field is available, it is of the form x.x.x.x
                if row[:6] == 'COMPND' and row[11:13] == 'EC':
                    self.ec_number = row[15:].strip()[:-1]

                # When a protein contains multiple chains, they will have a TER inbetween these.
                # The first time this is encountered, set "first_iter" to true (used in if statement just below).
                if row[:6] == 'TER   ':
                    first_iter = True

                if row[:6] == 'ATOM  ' and row[77:78] != 'H':
                    # If it is the first iteration, set current (start) chain to the field at row[21]. However, for some
                    # proteins this field is empty, in that case set it explicitly to A.
                    if first_iter:
                        curr_chain = row[21] or 'A'

                    res_seq = row[22:27]

                    # Found cases where res seq was '...,70,71A,71B,72,...'. Skip if not A or regular sequence.
                    if (res_seq[-1] != ' ' and res_seq[-1] != 'A'):
                        continue

                    res_seq = int(res_seq[:-1])

                    # Only consider positive residue sequence for now. Some proteins have a few negative residues before starting
                    # at 0, but these are (usually?) outliers. Can hope in the future these would get lost in the noise, but
                    # skip them for now.
                    if res_seq >= 0:
                        atom = row[13:15]
                        res = row[16:20]
                        chain = row[21] or 'A' # some PDB files don't have any chain at all
                        x,y,z = row[30:38].strip(), row[38:46].strip(), row[46:54].strip()

                        # For each residue, parse all the atoms and save in a sublist.
                        if ((old_res == -999 or old_res == res_seq)
                                and (res[0] == 'A' or res[0] == ' ')
                                and chain == curr_chain):
                            res_atoms.append(atom)
                            res_atoms_coords.append(tuple(map(float, (x,y,z))))

                        # Next residue or next chain
                        elif (res[0] == 'A' or res[0] == ' '):
                            chain_res_atoms.append(res_atoms)
                            chain_res_atoms_coords.append(res_atoms_coords)
                            res_atoms = [atom]
                            res_atoms_coords = [tuple(map(float, (x,y,z)))]


                        # Residue name considers alternative location
                        # Parse alpha carbons
                        if (atom == 'CA' and (res[0] == 'A' or res[0] == ' ')):
                            a_coords.append(tuple(map(float, (x,y,z))))

                        #  Parse beta carbons. Use alpha carbon for GLY.
                        if (atom == 'CB' and (res[0] == 'A' or res[0] == ' ') or
                                atom == 'CA' and (res == ' GLY' or res == 'AGLY')):
                            residues.append(res[1:])
                            res_sequence_number.append(str(res_seq) + res)
                            b_coords.append(tuple(map(float, (x,y,z))))
                            self.chain_length[chain] += 1

                        first_iter = False
                        old_res = res_seq
                        curr_chain = chain

        chain_res_atoms.append(res_atoms)
        chain_res_atoms_coords.append(res_atoms_coords)

        # Keep all residues, atoms and coordinates per chain
        for chain, amount in self.chain_length.items():
            self.residues[chain] = residues[:amount]
            self.res_sequence_number[chain] = res_sequence_number[:amount]
            self.a_coords[chain] = a_coords[:amount]
            self.b_coords[chain] = b_coords[:amount]
            self.res_atoms[chain] = chain_res_atoms[:amount]
            self.res_atoms_coords[chain] = [np.array(s) for s in chain_res_atoms_coords[:amount]] #chain_res_atoms_coords[:amount]
            self.length += amount


    def pairwise_distance_matrix(self, folder_path, out_name, c = 'A'):
        '''
        Fill a dictionary with pairwise interactions of amino acid residue pairs.
        Dictionary is saved as (k,v) with (AA pair, nr_of_contacts).
        This method uses `separation` which can be tuned accordingly.
        If the protein has multiple chains, nr_of_contacts will be all pairwise interactions
            internally within each chain, aggregated with all pairwise interactions between
            the chains.
        Accessed by self.pdm.
        '''
        if not self.length:
            print("Length of enzyme is 0, did you forget to parse the file?")
            raise RuntimeError()

        self.feature_pdm = {}
        self.aa = []
        
        for chain, _ in self.chain_length.items():
            if chain == c and self.exp_meth == 'X-RAY':

                # Calculate pairwise interactions separately for each chain, sum the result
                for i, c1 in enumerate(self.b_coords[chain]):
                    self.aa.append(self.res_sequence_number[chain][i])

                    for j, c2 in enumerate(self.b_coords[chain]):
                        if i <= j:
                            dist = round(self.__calc_dist(c1, c2), 1)
                            if self.feature_pdm.get(i) is None:
                                self.feature_pdm[i] = {}

                            if self.feature_pdm.get(j) is None:
                                self.feature_pdm[j] = {}

                            self.feature_pdm[i][j] = dist
                            self.feature_pdm[j][i] = dist

        # make data frame
        self.dist_mat = pd.DataFrame(self.feature_pdm)

        # change column and index names
        self.dist_mat.columns = self.aa
        self.dist_mat.index = self.aa
        self.dist_mat.reset_index()

        # save to file
        if self.exp_meth == 'X-RAY':
            self.dist_mat.to_csv(join(folder_path, '{}_dist_mat.tsv'.format(out_name)), sep='\t', index=True)



    def __calc_dist(self, c1, c2):
        '''
        Calculates the distance between two coordinates.
        Coordinates need shape as (x,y,z).
        '''
        dist = math.sqrt((c2[0] - c1[0]) ** 2 +
                         (c2[1] - c1[1]) ** 2 +
                         (c2[2] - c1[2]) ** 2)
        return dist




def main(infile, outfolder):
    '''
    Compute the distance matrix and save.
    '''
    assert exists(infile), 'Error, the file "{}" does not exist'.format(infile)
    assert exists(outfolder), 'Error, the folder "{}" does not exist'.format(outfolder)

    pdb_obj = PDB_Features()
    pdb_obj.parse_pdb(infile)
    pdb_obj.pairwise_distance_matrix(outfolder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Take a pdb file of as input. \n
    From this a distance matrix (between beta carbons) is computed
    and saved to file. The numbering (index and columns) reflects
    the amino acids in the structure, where some are often missing.''')

    parser.add_argument('--infile',help='A pdb file containing a protein structure.', metavar='')
    parser.add_argument('--outfolder',help='Folder to which output files should be written.', metavar='')

    args = parser.parse_args()

    infile = args.infile
    outfolder = args.outfolder

    main(infile, outfolder)
