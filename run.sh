#!/bin/bash
TAXON=specific_epithet
METRIC=f1_weighted
DIR_INPUT=/home/xandao/Imagens

for dataset in pr_dataset_features; do
    for image_size in 256 400 512; do
        for cnn in vgg16; do
            for color in GRAYSCALE RGB;	do
                for threshold in 5 10 20; do
                    echo ${cnn} ${size} ${threshold} ${color} ${METRIC}
                    python setup.py build_ext --inplace
                    python main.py -i ${DIR_INPUT}/${dataset}/${color}/${image_size}/${TAXON}/${threshold}/${cnn} -c 'DecisionTreeClassifier'
                done
            done
        done
    done
done
