import os

import numpy as np
from Bio import SeqIO

from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type=str, required=True,
                    help="")
parser.add_argument('-o', '--output', type=str, required=True,
                    help="")
parser.add_argument('-l', '--max_length', type=int, default=512,
                    help="")

# Define
AAS = 'ACDEFGHIKLMNPQRSTVWYX'

def get_combi( n_grams ):
    tmp = []
    for aa in AAS:
        for n in n_grams:
            tmp.append(str(aa+n))
    return tmp    

def get_ngram_index(word_length):
    n_grams = [aa for aa in AAS]
    if word_length > 1:
        for _ in range (word_length-1):
            n_grams = get_combi(n_grams)
    dict_ngram = {key: i for i, key in enumerate(n_grams)}            
    return dict_ngram

def get_seq_features(seq, word_length, dict_ngram_index):

    features = np.zeros((21**word_length,))
    length = len(seq)
    for ofset in range(word_length):
        list_of_ngrams = [ seq[i+ofset:i+word_length+ofset] for i in range(0,length-word_length-ofset,word_length)]
    for key in list_of_ngrams:
        features[dict_ngram_index[key]]+=1
    features/=length
    return features

def read_data(file, max_length = 512):
    data = {'id':[], 'ogt':[], 'seq':[], 'features_1': [], 'features_2': []}
    count = 0
    dict_ngram_index_1 = get_ngram_index(1)
    dict_ngram_index_2 = get_ngram_index(2)
    count = 0
    for i, rec in enumerate(SeqIO.parse(file, 'fasta')):
        if i%100!=0:
            continue
        count += 1
        data['id'].append(rec.id)
        data['ogt'].append(float(rec.description.split()[-1]))
        data['seq'].append(rec.seq)
        data['features_1'].append(get_seq_features(rec.seq, 1, dict_ngram_index_1))
        data['features_2'].append(get_seq_features(rec.seq, 2, dict_ngram_index_2))
        count += 1
    print("{} sequences loaded".format(count))
    return data

def get_pca(data, args):
    #for i, d in enumerate([data['features_1'], data['features_2']]):
    i = 0
    d = data['features_1']
    pca = PCA(n_components=2)
    pca.fit(d)
    X_pca = pca.transform(d)
    fig = plt.figure()
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data["ogt"], s = 1)
    plt.colorbar()
    plt.title('PCA plot of {}-gram frequency'.format(i+1))
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
   # legend2 = plt.legend(handles, ['Thermophiles', 'Mesophiles'], loc="upper right", title="Distributions")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.savefig("{}_{}_gram_pca.png".format(os.path.basename(args.input), i+1))
    
def get_tsne(data, args):
    perplx = [10, 30, 50, 100, 200]
    for p in perplx:
        tsne = TSNE(n_components=2, perplexity = p, n_jobs = -1)
        X_tsne =tsne.fit_transform(data['features_1'])
        plt.figure(figsize=[15,15])
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data["ogt"], s = 5)
        plt.colorbar(label="Temperature")
        plt.title('T-sne plot of {} perplexity {}'.format(os.path.basename(args.input),p))
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
       # legend2 = plt.legend(handles, ['Thermophiles', 'Mesophiles'], loc="upper right", title="Distributions")
        plt.xlabel('T-sne 1')
        plt.ylabel('T-sne 2')
        plt.savefig("{}_tsne_p{}.png".format(os.path.basename(args.input), p))
    
def get_umap(data, args):
    perplx = [40, 400]
    fi = [data["features_2"], data["features_2"]]
    for j, f in enumerate(fi):
        for p in perplx:
            reducer = umap.UMAP(n_neighbors=100, min_dist = 0.01)
            X_umap = reducer.fit_transform(f)
            plt.figure()
            scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data["ogt"], s = 1)
            plt.colorbar()
            plt.title('U-map plot of {} n-neighbors {}'.format(os.path.basename(args.input),p))
            handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
           # legend2 = plt.legend(handles, ['Thermophiles', 'Mesophiles'], loc="upper right", title="Distributions")
            plt.xlabel('Umap 1')
            plt.ylabel('Umap 2')
            plt.savefig("{}_{}_gram_umap_p{}.png".format(os.path.basename(args.input),j, p))


def main(args):
    data = read_data(args.input)
    get_pca(data, args)
    get_umap(data, args)
    #get_tsne(data, args)
    return 0

if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)