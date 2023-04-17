#!/bin/bash
TAXON=specific_epithet_trusted
METRIC=f1_weighted
DIR_INPUT=/home/xandao/Imagens
# DIR_INPUT=/media/kingston500/mestrado/dataset

for dataset in br_dataset_features; do
    for image_size in 512; do
        for cnn in vgg16; do
            for color in RGB GRAYSCALE; do
                for threshold in 20 10 5; do
#                    for classifier in DecisionTreeClassifier KNeighborsClassifier SVC; do
                    for classifier in MLPClassifier; do
                        python setup.py build_ext --inplace
                        if [ "regions_dataset_features" = "$dataset" ]; then
                            for region in Norte Nordeste Centro-Oeste Sul Sudeste; do
                                echo ${cnn} ${size} ${threshold} ${color} ${METRIC} ${classifier} ${region}
                                python main.py -i ${DIR_INPUT}/${dataset}/${color}/${image_size}/${TAXON}/${region}/${threshold}/${cnn} -c ${classifier} -p
                            done
                        else
                            echo ${cnn} ${size} ${threshold} ${color} ${METRIC} ${classifier}
                            python main.py -i ${DIR_INPUT}/${dataset}/${color}/${image_size}/${TAXON}/${threshold}/${cnn} -c ${classifier}
                        fi
                    done
                done
            done
        done
    done
done
