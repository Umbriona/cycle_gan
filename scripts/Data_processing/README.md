#### Download enzyme datasets from Zenodo  to `../data`
https://zenodo.org/record/2539114/files/brenda_sequences_20180109.fasta  
https://zenodo.org/record/2539114/files/enzyme_ogt_topt.tsv

#### Removed sequences
(1) containing non-standard amino acids  
(2) longer than 2000.   
(3) shorter than 100  
This produces a fasta file `cleaned_seqs_1.0.fasta`

#### Remove redundant sequences with cd-hit with an identity cut-off of 95%

#### Remove sequences in Topt dataset
