#!/bin/bash
TAXON=specific_epithet_trusted
METRIC=f1_weighted
DIR_INPUT=/home/none/Imagens
# DIR_INPUT=/media/kingston500/mestrado/dataset

for dataset in pr_dataset_features; do
    for image_size in 512 400 256; do
        for cnn in lbp.txt surf64.txt; do
            for color in GRAYSCALE; do
                for threshold in 20 10 5; do
                    for contrast in 1.8 1.5 1.2; do
                        INPUT=${DIR_INPUT}/${dataset}_CONTRAST_${contrast}/${color}/${TAXON}/${image_size}
    #                    for classifier in DecisionTreeClassifier; do
    #                    for classifier in KNeighborsClassifierClassifier; do
    #                    for classifier in RandomForestClassifier; do
    #                    for classifier in MLPClassifier; do
    #                    for classifier in SVC; do
                        for classifier in DecisionTreeClassifier KNeighborsClassifier RandomForestClassifier MLPClassifier SVC; do
                            python setup.py build_ext --inplace
                            if [ "regions_dataset_features" = "$dataset" ]; then
                                for region in Norte Nordeste Centro-Oeste Sul Sudeste; do
                                    echo ${cnn} ${size} ${threshold} ${color} ${METRIC} ${classifier} ${region}
                                    echo ${INPUT}
                                    python main.py -i ${INPUT}/${region}/${threshold}/${cnn} -c ${classifier}
                                done
                            else
                                echo ${cnn} ${size} ${threshold} ${color} ${METRIC} ${classifier}
                                echo ${INPUT}
                                python main.py -i ${INPUT}/${threshold}/${cnn} -c ${classifier}
                            fi
                        done
                    done
                done
            done
        done
    done
done
