#!/bin/bash
TAXON=specific_epithet_trusted
METRIC=f1_weighted
DIR_INPUT=/home/xandao/Imagens
# DIR_INPUT=/media/kingston500/mestrado/dataset

for dataset in regions_dataset_features; do
    for image_size in 256 400 512; do
        for cnn in vgg16; do
            for color in GRAYSCALE RGB;	do
                for threshold in 5 10 20; do
                    for classifier in DecisionTreeClassifier KNeighborsClassifier RandomForestClassifier SVC; do
                        python setup.py build_ext --inplace
                        if [ "regions_dataset_features" = "$dataset" ]; then
                            for region in Norte Nordeste Centro-Oeste Sul Sudeste; do
                                echo ${cnn} ${size} ${threshold} ${color} ${METRIC} ${classifier} ${region}
                                python main.py -i ${DIR_INPUT}/${dataset}/${color}/${image_size}/${TAXON}/${region}/${threshold}/${cnn} -c ${classifier} -p
                            done
                        else
                            echo ${cnn} ${size} ${threshold} ${color} ${METRIC} ${classifier}
                            python main.py -i ${DIR_INPUT}/${dataset}/${color}/${image_size}/${TAXON}/${threshold}/${cnn} -c ${classifier} -p
                        fi
                    done
                done
            done
        done
    done
done
