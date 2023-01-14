#!/bin/bash
PY=~/miniconda3/bin/python
TAXON=specific_epithet
METRIC=f1_weighted
DIR_FEATURES=../dataset_gimp/imagens_george/features
DIR_LABEL=../dataset_gimp/imagens_george/imagens

for img_size in 256 400 512; do
	for cnn in mobilenetv2 resnet50v2 vgg16; do
		for color in grayscale; do
		    for patch in 3; do
                for threshold in 5; do
                    echo ${cnn} ${img_size} ${threshold} ${color} ${patch} ${METRIC}
                    ${PY} main.py -i ${DIR_FEATURES}/${color}/segmented_unet/${img_size}/patch=${patch}/${TAXON}/${threshold}/${cnn}/horizontal -l ${DIR_LABEL}/${color}/${TAXON}/${img_size}/${threshold}/label2.txt -m ${METRIC}
                done
			done
		done
	done
done
