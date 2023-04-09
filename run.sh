#!/bin/bash
TAXON=specific_epithet_trusted
METRIC=f1_weighted
DIR_INPUT=/home/xandao/Imagens

for dataset in pr_dataset_features; do
    for image_size in 256 400 512; do
        for cnn in vgg16; do
            for color in GRAYSCALE RGB;	do
                for threshold in 5 10 20; do
                    for classifier in DecisionTreeClassifier KNeighborsClassifier MLPClassifier RandomForestClassifier         SVC; do
                        echo ${cnn} ${size} ${threshold} ${color} ${METRIC}
                        python setup.py build_ext --inplace
                        # /home/xandao/Imagens/pr_dataset_features/RGB/256/specific_epithet_trusted/20/vgg16
                        python main.py -i ${DIR_INPUT}/${dataset}/${color}/${image_size}/${TAXON}/${threshold}/${cnn} -c ${classifier} -p
                    done
                done
            done
        done
    done
done
