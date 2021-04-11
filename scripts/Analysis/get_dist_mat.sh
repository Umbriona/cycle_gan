#! /bin/bash

$path_ali=../results/OG_ali
$path_dist=../results/OG_dist_mat

NAME_LIST=

for NAME in ${NAME_TAX[@]}
do
        for ID in ${id[@]}
        do
                echo "Creating data set id" $ID "from" $NAME "/n"
                ./usearch -cluster_fast $FILE_PATH$NAME$FILE_EXT -id $ID -centroids $FILE_PATH$NAME"_rn_"$ID$FILE_EXT
        done
done