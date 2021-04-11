import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type = str)
parser.add_argument('-o', '--output', type = str)

def plot_histogram(args, list_of_means):
    
    plt.figure(figsize=[15,15])


def main(args):
    list_of_means = []
    for file in args.input:
        with open(file, "r") as f:
            string = f.read()
        dict_dist ={}
        for i in string.split('\n')[1:-1]:
            l = i.split(" ")
            l_ = []
            for j in l[1:]:
                try:
                    l_.append(float(j))
                except:
                    pass

        dict_dist[l[0]] = l_
        df_dist = pd.DataFrame(dict_dist, index=dict_dist.keys())
        list_of_means.extend( df_dist.mean().values )
    
    
    
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)